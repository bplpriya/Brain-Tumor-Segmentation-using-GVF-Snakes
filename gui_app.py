"""
gui_app.py
----------
Brain Tumour Segmentation — Interactive GUI
==========================================

Features:
  • Upload any brain MRI image (TIF / PNG / JPG)
  • Auto-detect tumour using GVF snake
  • Side-by-side: Original | GVF result | Traditional result
  • Real-time snake animation with artistic rendering
  • Dice score display (if ground truth mask provided)
  • Dataset browser (LGG Kaggle dataset)

Usage:
    python gui_app.py
    python gui_app.py --dataset_path /path/to/lgg-mri-segmentation

Dependencies: numpy, scipy, matplotlib, scikit-image, Pillow
NOTE: This GUI uses matplotlib figures embedded in a simple HTML-like
      layout rendered via matplotlib (no tkinter required).
      For interactive use, run from terminal (not headless server).
"""

import sys, os, argparse, glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")        # Works on desktop; falls back if unavailable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_mri   import generate_brain_mri
from snake_traditional import run_traditional_snake, init_circle
from gvf             import run_gvf_snake
from metrics         import contour_to_mask, dice_coefficient
from snake_art       import draw_snake_on_axes


# ─────────────────────────────────────────────────────────────────────────────
# Colour theme
# ─────────────────────────────────────────────────────────────────────────────

BG        = "#080810"
PANEL_BG  = "#111120"
ACCENT    = "#00FF88"
TEXT      = "#DDDDEE"
BTN_BG    = "#1A2040"
BTN_HOVER = "#2A3060"
WARN      = "#FF4444"
GOLD      = "#FFD700"


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str, size: int = 256) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float64)
    return arr / (arr.max() + 1e-10)


def load_mask(path: str, size: int = 256) -> np.ndarray | None:
    """Try to find a corresponding *_mask.* file next to the image."""
    base, ext = os.path.splitext(path)
    candidates = [base + "_mask" + ext, base + "_mask.tif",
                  base + "_mask.png", base + "_mask.jpg"]
    for c in candidates:
        if os.path.isfile(c):
            mask = Image.open(c).convert("L").resize(
                (size, size), Image.NEAREST)
            return np.array(mask, dtype=bool)
    return None


def auto_detect_center(image: np.ndarray, gt_mask: np.ndarray = None):
    """
    Detect tumour centre.

    Priority order:
      1. If gt_mask is provided → use its centroid (most accurate)
      2. Otherwise → use local-variance map (tumours have heterogeneous texture)
         combined with excluding the skull strip (bright outer ring)
    """
    from scipy.ndimage import label, center_of_mass, uniform_filter

    h, w = image.shape

    # ── Case 1: ground truth mask available ──────────────────────────────────
    if gt_mask is not None and gt_mask.any():
        ys, xs = np.where(gt_mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        # radius = bounding circle of the mask
        dists = np.sqrt((ys - cy)**2 + (xs - cx)**2)
        r = max(12, int(dists.max() * 0.9))
        return (cy, cx), r

    # ── Case 2: no mask — smarter heuristic for real brain MRI ───────────────
    # Step 1: Remove skull (brightest ~10% of pixels are skull/fat)
    skull_thresh = np.percentile(image, 90)
    brain_mask   = image < skull_thresh

    # Step 2: Local variance map — tumours have irregular texture
    mean_local = uniform_filter(image, size=15)
    mean_sq    = uniform_filter(image**2, size=15)
    variance   = np.clip(mean_sq - mean_local**2, 0, None)

    # Step 3: Intermediate brightness band (tumours are not darkest or brightest)
    lo = np.percentile(image, 30)
    hi = np.percentile(image, 85)
    mid_mask = (image >= lo) & (image <= hi) & brain_mask

    # Score = variance * mid_brightness_mask
    score = variance * mid_mask.astype(float)

    # Step 4: Threshold score and find largest blob
    thresh = np.percentile(score[score > 0], 80) if score.max() > 0 else 0
    hot = score > thresh
    labeled, n = label(hot)
    if n == 0:
        # fallback: image centre
        return (h // 2, w // 2), 30

    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    biggest = np.argmax(sizes) + 1
    region  = labeled == biggest
    cy, cx  = center_of_mass(region)
    ys2, xs2 = np.where(region)
    r = max(15, int(np.sqrt(((ys2 - cy)**2 + (xs2 - cx)**2).mean()) * 1.4))
    # Cap radius so we don't initialise off the image
    r = min(r, min(h, w) // 4)
    return (int(cy), int(cx)), r


# ─────────────────────────────────────────────────────────────────────────────
# Core segmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationPipeline:
    def __init__(self, image, gt_mask=None,
                 alpha=0.015, beta=0.1, mu=0.25, sigma=2.5,
                 snake_iter=3000, gvf_iter=300):
        self.image      = image
        self.gt_mask    = gt_mask
        self.alpha      = alpha
        self.beta       = beta
        self.mu         = mu
        self.sigma      = sigma
        self.snake_iter = snake_iter
        self.gvf_iter   = gvf_iter

        # Use mask centroid if available, otherwise smart heuristic
        (self.cy, self.cx), self.radius = auto_detect_center(image, gt_mask)
        print(f"  [Pipeline] Centre=({self.cx},{self.cy}), radius={self.radius}")

    def run_gvf(self, init_radius_extra=10, n_points=120, store_every=60):
        init_r = self.radius + init_radius_extra
        # Clamp to image bounds
        h, w = self.image.shape
        init_r = min(init_r, min(self.cx, w - self.cx, self.cy, h - self.cy) - 5)
        init_r = max(init_r, 10)
        x0, y0 = init_circle((self.cx, self.cy), init_r, n_points)
        gvf_x, gvf_y, gvf_hist, u, v, edge_map = run_gvf_snake(
            self.image, x0, y0,
            alpha=self.alpha, beta=self.beta,
            gamma=2.0,          # must be 2.0 for true GVF matrix snake
            mu=self.mu, sigma=self.sigma, kappa=0,
            gvf_iter=self.gvf_iter, snake_iter=self.snake_iter,
            dt=0.2, store_every=store_every,
            force_scale=5.0,    # lower than default for real images
        )
        self.gvf_x, self.gvf_y   = gvf_x, gvf_y
        self.gvf_hist             = gvf_hist
        self.u, self.v, self.edge_map = u, v, edge_map
        self.gvf_mask             = contour_to_mask(gvf_x, gvf_y, self.image.shape)
        self.gvf_dice = dice_coefficient(self.gvf_mask, self.gt_mask) \
            if self.gt_mask is not None else None
        return gvf_x, gvf_y, gvf_hist

    def run_traditional(self, init_radius_extra=10, n_points=120, store_every=60):
        init_r = self.radius + init_radius_extra
        h, w = self.image.shape
        init_r = min(init_r, min(self.cx, w - self.cx, self.cy, h - self.cy) - 5)
        init_r = max(init_r, 10)
        x0, y0 = init_circle((self.cx, self.cy), init_r, n_points)
        trad_x, trad_y, trad_hist = run_traditional_snake(
            self.image, x0, y0,
            alpha=self.alpha, beta=self.beta, gamma=0.01,
            sigma=self.sigma, w_line=0.0, w_edge=1.0,
            n_iter=self.snake_iter, store_every=store_every,
        )
        self.trad_x, self.trad_y = trad_x, trad_y
        self.trad_hist            = trad_hist
        self.trad_mask            = contour_to_mask(trad_x, trad_y, self.image.shape)
        self.trad_dice = dice_coefficient(self.trad_mask, self.gt_mask) \
            if self.gt_mask is not None else None
        return trad_x, trad_y, trad_hist


# ─────────────────────────────────────────────────────────────────────────────
# GUI class
# ─────────────────────────────────────────────────────────────────────────────

class BrainTumourGUI:
    """
    Interactive matplotlib-based GUI.

    Layout:
    ┌──────────────────────────────────────────┬──────────────┐
    │   MRI + GVF result + Traditional result  │  Control     │
    │        (main display panel)              │  Panel       │
    ├──────────────────────────────────────────┤              │
    │   GVF field  |  Metrics bar  |  Status   │              │
    └──────────────────────────────────────────┴──────────────┘
    """

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self.image        = None
        self.gt_mask      = None
        self.pipeline     = None
        self.current_file = None
        self.dataset_files= []
        self.ds_index     = 0
        self._anim_frame  = 0
        self._anim_hist   = []
        self._animating   = False

        if dataset_path:
            self._scan_dataset(dataset_path)

        self._build_layout()
        self._show_welcome()

    # ── dataset scanner ───────────────────────────────────────────────────────

    def _scan_dataset(self, path):
        masks = glob.glob(os.path.join(path, "**", "*_mask.*"), recursive=True)
        self.dataset_files = []
        for mp in sorted(masks):
            ip = mp.replace("_mask", "")
            if os.path.isfile(ip):
                mask_arr = np.array(Image.open(mp).convert("L"))
                if mask_arr.max() > 0:   # only tumour-positive slices
                    self.dataset_files.append((ip, mp))
        print(f"[GUI] Found {len(self.dataset_files)} tumour-positive slices")

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_layout(self):
        self.fig = plt.figure(figsize=(18, 10), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "Brain Tumour Segmentation — Active Contour GUI")

        # Main grid: left (displays) | right (controls)
        gs_root = gridspec.GridSpec(1, 2, figure=self.fig,
                                    width_ratios=[3.2, 1],
                                    wspace=0.05)

        # Left sub-grid: top row (3 images) + bottom row (field/metrics/status)
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=gs_root[0],
            hspace=0.08, wspace=0.06,
            height_ratios=[2, 1])

        # Display axes
        self.ax_orig   = self.fig.add_subplot(gs_left[0, 0])
        self.ax_gvf    = self.fig.add_subplot(gs_left[0, 1])
        self.ax_trad   = self.fig.add_subplot(gs_left[0, 2])
        self.ax_field  = self.fig.add_subplot(gs_left[1, 0])
        self.ax_metric = self.fig.add_subplot(gs_left[1, 1])
        self.ax_status = self.fig.add_subplot(gs_left[1, 2])

        # Right control panel
        gs_ctrl = gridspec.GridSpecFromSubplotSpec(
            12, 1, subplot_spec=gs_root[1], hspace=0.4)
        self.ax_ctrl_title = self.fig.add_subplot(gs_ctrl[0])
        self.ax_btn_load   = self.fig.add_subplot(gs_ctrl[1])
        self.ax_btn_synth  = self.fig.add_subplot(gs_ctrl[2])
        self.ax_btn_prev   = self.fig.add_subplot(gs_ctrl[3])
        self.ax_btn_next   = self.fig.add_subplot(gs_ctrl[4])
        self.ax_btn_run    = self.fig.add_subplot(gs_ctrl[5])
        self.ax_btn_anim   = self.fig.add_subplot(gs_ctrl[6])
        self.ax_btn_save   = self.fig.add_subplot(gs_ctrl[7])
        self.ax_sl_alpha   = self.fig.add_subplot(gs_ctrl[8])
        self.ax_sl_mu      = self.fig.add_subplot(gs_ctrl[9])
        self.ax_sl_iter    = self.fig.add_subplot(gs_ctrl[10])
        self.ax_info       = self.fig.add_subplot(gs_ctrl[11])

        # Style all axes
        for ax in [self.ax_orig, self.ax_gvf, self.ax_trad,
                   self.ax_field, self.ax_metric, self.ax_status,
                   self.ax_ctrl_title, self.ax_info]:
            ax.set_facecolor(PANEL_BG)
            ax.tick_params(colors=TEXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#334")

        # Buttons
        self.btn_load  = Button(self.ax_btn_load,  "📁 Load Image",
                                color=BTN_BG, hovercolor=BTN_HOVER)
        self.btn_synth = Button(self.ax_btn_synth, "🧠 Synthetic MRI",
                                color=BTN_BG, hovercolor=BTN_HOVER)
        self.btn_prev  = Button(self.ax_btn_prev,  "◀ Prev Slice",
                                color=BTN_BG, hovercolor=BTN_HOVER)
        self.btn_next  = Button(self.ax_btn_next,  "▶ Next Slice",
                                color=BTN_BG, hovercolor=BTN_HOVER)
        self.btn_run   = Button(self.ax_btn_run,   "▶ Run Segmentation",
                                color="#1A4030", hovercolor="#2A6048")
        self.btn_anim  = Button(self.ax_btn_anim,  "🎬 Save Animation",
                                color=BTN_BG, hovercolor=BTN_HOVER)
        self.btn_save  = Button(self.ax_btn_save,  "💾 Save Results",
                                color=BTN_BG, hovercolor=BTN_HOVER)

        for btn in [self.btn_load, self.btn_synth, self.btn_prev,
                    self.btn_next, self.btn_run, self.btn_anim, self.btn_save]:
            btn.label.set_color(TEXT)
            btn.label.set_fontsize(9)

        # Sliders
        self.sl_alpha = Slider(self.ax_sl_alpha, "α  (elasticity)",
                               0.001, 0.1, valinit=0.015, color=ACCENT)
        self.sl_mu    = Slider(self.ax_sl_mu,    "μ  (GVF diffusion)",
                               0.05, 0.5, valinit=0.25, color="#00CCFF")
        self.sl_iter  = Slider(self.ax_sl_iter,  "Iterations",
                               500, 8000, valinit=3000, valstep=500,
                               color="#FF8844")
        for sl in [self.sl_alpha, self.sl_mu, self.sl_iter]:
            sl.label.set_color(TEXT); sl.label.set_fontsize(8)
            sl.valtext.set_color(ACCENT)

        # Callbacks
        self.btn_load.on_clicked(self._cb_load)
        self.btn_synth.on_clicked(self._cb_synthetic)
        self.btn_prev.on_clicked(self._cb_prev)
        self.btn_next.on_clicked(self._cb_next)
        self.btn_run.on_clicked(self._cb_run)
        self.btn_anim.on_clicked(self._cb_save_anim)
        self.btn_save.on_clicked(self._cb_save_results)

        # Control panel title
        self.ax_ctrl_title.set_facecolor("#0A0A20")
        self.ax_ctrl_title.text(0.5, 0.5,
            "🐍  GVF Brain Tumour\n   Segmentation",
            transform=self.ax_ctrl_title.transAxes,
            ha="center", va="center", color=ACCENT,
            fontsize=11, fontweight="bold")
        self.ax_ctrl_title.axis("off")

    # ── welcome screen ────────────────────────────────────────────────────────

    def _show_welcome(self):
        for ax in [self.ax_orig, self.ax_gvf, self.ax_trad,
                   self.ax_field, self.ax_metric, self.ax_status]:
            ax.cla(); ax.set_facecolor(PANEL_BG); ax.axis("off")

        welcome_text = (
            "Welcome to the GVF Brain Tumour Segmentation Demo\n\n"
            "To get started:\n"
            "  1. Click '📁 Load Image' to upload a brain MRI\n"
            "  2. Or click '🧠 Synthetic MRI' to use generated data\n"
            "  3. If you have the LGG dataset, use ◀ / ▶ to browse slices\n"
            "  4. Click '▶ Run Segmentation' to detect the tumour\n\n"
            "The GVF snake will slither to the tumour boundary!"
        )
        self.ax_orig.text(0.5, 0.5, welcome_text,
                          transform=self.ax_orig.transAxes,
                          ha="center", va="center", color=TEXT,
                          fontsize=10, linespacing=1.8,
                          bbox=dict(facecolor="#111130", alpha=0.8,
                                    boxstyle="round,pad=0.8",
                                    edgecolor=ACCENT))
        self._status(f"Dataset: {len(self.dataset_files)} slices available" 
                     if self.dataset_files else "No dataset loaded")
        self.fig.canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cb_load(self, event):
        """Load image via file dialog."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            path = filedialog.askopenfilename(
                title="Select Brain MRI Image",
                filetypes=[("Image files",
                            "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                           ("All files", "*.*")])
            root.destroy()
            if path:
                self._load_image_file(path)
        except ImportError:
            # tkinter not available — show path prompt
            self._status("tkinter unavailable. Use dataset browser or Synthetic MRI.")

    def _cb_synthetic(self, event):
        self._status("Generating synthetic brain MRI...")
        self.fig.canvas.draw_idle()
        image, gt_mask, (tcy, tcx), (try_, trx) = generate_brain_mri(
            size=256, seed=np.random.randint(0, 999))
        self.image   = image
        self.gt_mask = gt_mask
        self.current_file = "Synthetic MRI"
        self._display_input()
        self._status("Synthetic MRI loaded. Click ▶ Run Segmentation.")

    def _cb_prev(self, event):
        if not self.dataset_files:
            self._status("No dataset. Load LGG dataset first."); return
        self.ds_index = (self.ds_index - 1) % len(self.dataset_files)
        self._load_dataset_slice()

    def _cb_next(self, event):
        if not self.dataset_files:
            self._status("No dataset. Load LGG dataset first."); return
        self.ds_index = (self.ds_index + 1) % len(self.dataset_files)
        self._load_dataset_slice()

    def _cb_run(self, event):
        if self.image is None:
            self._status("⚠ No image loaded!"); return
        self._run_segmentation()

    def _cb_save_anim(self, event):
        if self.pipeline is None or not hasattr(self.pipeline, "gvf_hist"):
            self._status("Run segmentation first!"); return
        os.makedirs("gui_outputs", exist_ok=True)
        from snake_art import animate_snake
        self._status("Saving animation...")
        self.fig.canvas.draw_idle()
        animate_snake(self.image, self.gt_mask, self.pipeline.gvf_hist,
                      title="GVF Snake — Brain Tumour Segmentation",
                      save_path="gui_outputs/snake_animation.gif", fps=10)
        self._status("Animation saved → gui_outputs/snake_animation.gif")

    def _cb_save_results(self, event):
        if self.pipeline is None: self._status("Run first!"); return
        os.makedirs("gui_outputs", exist_ok=True)
        self.fig.savefig("gui_outputs/gui_result.png", dpi=150,
                          bbox_inches="tight", facecolor=BG)
        self._status("Saved → gui_outputs/gui_result.png")

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_image_file(self, path):
        self._status(f"Loading: {os.path.basename(path)}")
        self.image      = load_image(path)
        self.gt_mask    = load_mask(path)
        self.current_file = path
        self._display_input()
        has_mask = "✓ Mask found" if self.gt_mask is not None else "No mask"
        self._status(f"Loaded {os.path.basename(path)} ({has_mask}). "
                     "Click ▶ Run Segmentation.")

    def _load_dataset_slice(self):
        ip, mp = self.dataset_files[self.ds_index]
        self._status(f"[{self.ds_index+1}/{len(self.dataset_files)}] "
                     f"{os.path.basename(ip)}")
        self.image = load_image(ip)
        mask_arr   = Image.open(mp).convert("L").resize((256,256), Image.NEAREST)
        self.gt_mask = np.array(mask_arr, dtype=bool)
        self.current_file = ip
        self._display_input()

    # ── display helpers ───────────────────────────────────────────────────────

    def _display_input(self):
        self.ax_orig.cla(); self.ax_orig.set_facecolor(PANEL_BG)
        self.ax_orig.imshow(self.image, cmap="gray", vmin=0, vmax=1)
        if self.gt_mask is not None:
            ov = np.zeros((*self.image.shape, 4))
            ov[self.gt_mask] = [1.0, 0.2, 0.2, 0.45]
            self.ax_orig.imshow(ov)
        self.ax_orig.set_title("Input MRI\n(red = GT tumour)",
                                color=TEXT, fontsize=10)
        self.ax_orig.axis("off")

        # Clear result panels
        for ax, ttl in [(self.ax_gvf, "GVF Snake Result"),
                        (self.ax_trad, "Traditional Snake")]:
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ax.text(0.5, 0.5, f"Run segmentation\nto see {ttl}",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#888", fontsize=9)
            ax.set_title(ttl, color=TEXT, fontsize=10); ax.axis("off")
        self.ax_field.cla();  self.ax_field.set_facecolor(PANEL_BG)
        self.ax_metric.cla(); self.ax_metric.set_facecolor(PANEL_BG)
        for ax in [self.ax_field, self.ax_metric]:
            ax.axis("off")
        self.fig.canvas.draw_idle()

    def _status(self, msg):
        self.ax_status.cla(); self.ax_status.set_facecolor(PANEL_BG)
        self.ax_status.text(0.5, 0.5, msg, transform=self.ax_status.transAxes,
                             ha="center", va="center", color=TEXT,
                             fontsize=9, wrap=True)
        self.ax_status.axis("off")
        self.fig.canvas.draw_idle()

    # ── segmentation ──────────────────────────────────────────────────────────

    def _run_segmentation(self):
        alpha = self.sl_alpha.val
        mu    = self.sl_mu.val
        iters = int(self.sl_iter.val)

        self._status("Detecting tumour centre...")
        self.fig.canvas.draw_idle()

        self.pipeline = SegmentationPipeline(
            self.image, self.gt_mask,
            alpha=alpha, mu=mu, snake_iter=iters, gvf_iter=250)

        cx, cy, r = self.pipeline.cx, self.pipeline.cy, self.pipeline.radius
        self._status(f"Centre=({cx},{cy}) r={r}px | Computing GVF field...")
        self.fig.canvas.draw_idle()

        # Use +10px extra for both (same distance comparison)
        self.pipeline.run_gvf(init_radius_extra=10)

        self._status(f"GVF done. Running traditional snake...")
        self.fig.canvas.draw_idle()
        self.pipeline.run_traditional(init_radius_extra=10)

        self._display_results()
        self._display_field()
        self._display_metrics()

        gd = self.pipeline.gvf_dice
        td = self.pipeline.trad_dice
        if gd is not None:
            dice_str = f"GVF Dice={gd:.4f}  |  Trad Dice={td:.4f}"
        else:
            dice_str = "No GT mask — load a *_mask.tif file for Dice scores"
        self._status(f"✓ Done!  {dice_str}")

    def _display_results(self):
        pl = self.pipeline
        ov = np.zeros((*self.image.shape, 4))
        if self.gt_mask is not None:
            ov[self.gt_mask] = [1.0, 0.2, 0.2, 0.30]

        for ax, xf, yf, prog, ttl, dice in [
            (self.ax_gvf,  pl.gvf_x,  pl.gvf_y,  0.95,
             f"GVF Snake ({'Dice={:.4f}'.format(pl.gvf_dice) if pl.gvf_dice else 'no GT'})",
             pl.gvf_dice),
            (self.ax_trad, pl.trad_x, pl.trad_y, 0.15,
             f"Traditional ({'Dice={:.4f}'.format(pl.trad_dice) if pl.trad_dice else 'no GT'})",
             pl.trad_dice),
        ]:
            ax.cla(); ax.set_facecolor(PANEL_BG)
            ax.imshow(self.image, cmap="gray", vmin=0, vmax=1, alpha=0.80)
            ax.imshow(ov, zorder=1)
            draw_snake_on_axes(ax, xf, yf, progress=prog,
                               show_scales=True, show_head=True, lw_base=3.0)
            if dice is not None:
                col = ACCENT if prog > 0.5 else WARN
                ax.text(4, self.image.shape[0]-5, f"Dice={dice:.4f}",
                        color=col, fontsize=10, fontweight="bold", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.5,
                                  boxstyle="round,pad=0.25", edgecolor="none"))
            ax.set_title(ttl, color=TEXT, fontsize=10)
            ax.axis("off")

    def _display_field(self):
        pl = self.pipeline
        ax = self.ax_field
        ax.cla(); ax.set_facecolor(PANEL_BG)
        ax.imshow(self.image, cmap="gray", alpha=0.55, vmin=0, vmax=1)
        step = 12
        h, w = self.image.shape
        ys = np.arange(step//2, h, step)
        xs = np.arange(step//2, w, step)
        X, Y = np.meshgrid(xs, ys)
        U = pl.u[ys[:, None], xs[None, :]]
        V = pl.v[ys[:, None], xs[None, :]]
        mag = np.sqrt(U**2 + V**2)
        # Normalise scale to image units (not data units)
        ax.quiver(X, Y, U, -V, mag, cmap="plasma",
                  units="xy", scale=0.3, width=0.8,
                  headwidth=3, headlength=4, alpha=0.92)
        ax.set_xlim(0, w); ax.set_ylim(h, 0)
        ax.set_title("GVF Force Field", color=TEXT, fontsize=9)
        ax.axis("off")

    def _display_metrics(self):
        pl = self.pipeline
        ax = self.ax_metric
        ax.cla(); ax.set_facecolor(PANEL_BG)

        if pl.gvf_dice is None:
            ax.text(0.5, 0.5, "Upload image with\nground truth mask\nfor Dice score",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#888", fontsize=9)
            ax.axis("off"); return

        from metrics import precision_recall
        gd = pl.gvf_dice;  gp, gr = precision_recall(pl.gvf_mask, self.gt_mask)
        td = pl.trad_dice; tp, tr = precision_recall(pl.trad_mask, self.gt_mask)

        x = np.arange(3); w = 0.32
        vals_t = [td, tp, tr]; vals_g = [gd, gp, gr]
        b1 = ax.bar(x-w/2, vals_t, w, color="#FF6600", alpha=0.85, label="Trad")
        b2 = ax.bar(x+w/2, vals_g, w, color="#00CC66", alpha=0.85, label="GVF")
        for bar, v in zip(list(b1)+list(b2), vals_t+vals_g):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{v:.3f}", ha="center", va="bottom",
                    color="white", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Dice","Prec","Rec"], color=TEXT, fontsize=8)
        ax.set_ylim(0, 1.25)
        ax.tick_params(colors=TEXT, labelsize=7)
        ax.spines[:].set_color("#334")
        ax.set_facecolor(PANEL_BG)
        ax.set_title("Metrics", color=TEXT, fontsize=9)
        ax.legend(fontsize=7, facecolor="#111", edgecolor="#444",
                  labelcolor="white")

    # ── show ─────────────────────────────────────────────────────────────────

    def show(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Brain Tumour Segmentation GUI")
    ap.add_argument("--dataset_path", default=None,
                    help="Path to LGG MRI segmentation dataset root")
    ap.add_argument("--image", default=None,
                    help="Directly open a specific MRI image")
    args = ap.parse_args()

    gui = BrainTumourGUI(dataset_path=args.dataset_path)

    if args.image:
        gui._load_image_file(args.image)
    elif not args.dataset_path:
        # Auto-load synthetic if no dataset provided
        from synthetic_mri import generate_brain_mri
        image, gt_mask, _, _ = generate_brain_mri(size=256)
        gui.image    = image
        gui.gt_mask  = gt_mask
        gui.current_file = "Synthetic MRI (auto)"
        gui._display_input()
        gui._status("Synthetic MRI loaded. Click ▶ Run Segmentation to begin.")

    gui.show()


if __name__ == "__main__":
    main()
