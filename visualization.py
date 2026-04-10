"""
visualization.py
----------------
All plotting utilities for the Brain Tumour Segmentation project.
Produces publication-quality figures and an animated GIF.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _close_contour(x, y):
    """Append first point to close the contour loop."""
    return np.append(x, x[0]), np.append(y, y[0])


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Synthetic MRI + Ground Truth
# ─────────────────────────────────────────────────────────────────────────────

def plot_mri_and_gt(image, gt_mask, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             facecolor="#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Synthetic T1 Brain MRI", color="white", fontsize=13, pad=8)
    axes[0].axis("off")

    overlay = np.zeros((*image.shape, 4))
    overlay[gt_mask] = [1.0, 0.2, 0.2, 0.55]   # red semi-transparent
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(overlay)
    axes[1].set_title("Ground Truth Tumour Mask", color="white", fontsize=13, pad=8)
    axes[1].axis("off")

    red_patch = mpatches.Patch(color=(1, 0.2, 0.2, 0.7), label="Tumour Region")
    axes[1].legend(handles=[red_patch], loc="lower right",
                   facecolor="#1a1a1a", edgecolor="white",
                   labelcolor="white", fontsize=10)

    fig.suptitle("Brain Tumour Segmentation — Dataset",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – GVF Vector Field (artistic visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def plot_gvf_field(image, u, v, edge_map,
                   step=10, scale=18, save_path=None):
    """
    Colour-coded GVF quiver plot – force magnitude drives colour.
    This is the 'artistic' demo component described in the checkpoint.
    """
    h, w = image.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    X, Y = np.meshgrid(xs, ys)

    U = u[ys[:, None], xs[None, :]]
    V = v[ys[:, None], xs[None, :]]
    mag = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#050510")
    ax.set_facecolor("#050510")

    ax.imshow(image, cmap="gray", alpha=0.55, vmin=0, vmax=1)

    q = ax.quiver(X, Y, U, -V,          # flip V for image coords
                  mag,
                  cmap="plasma",
                  scale=scale,
                  scale_units="xy",
                  width=0.003,
                  alpha=0.85)

    cbar = fig.colorbar(q, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("GVF Force Magnitude", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Gradient Vector Flow (GVF) Field\n"
                 "Colour = Force Magnitude | Arrows = Force Direction",
                 color="white", fontsize=12, pad=10)
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Side-by-side: Traditional vs GVF Snake
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(image, gt_mask,
                    trad_history, gvf_history,
                    trad_final, gvf_final,
                    trad_dice, gvf_dice,
                    save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5),
                             facecolor="#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    titles = [
        f"Traditional Snake (Kass 1988)\nDice = {trad_dice:.4f}",
        f"GVF Snake (Xu & Prince 1998)\nDice = {gvf_dice:.4f}",
    ]
    histories = [trad_history, gvf_history]
    finals = [trad_final, gvf_final]
    colours = ["#FFD700", "#00FF88"]   # gold for traditional, green for GVF

    for ax, title, hist, (xf, yf), col in zip(
            axes, titles, histories, finals, colours):

        ax.imshow(image, cmap="gray", vmin=0, vmax=1)

        # GT overlay
        overlay = np.zeros((*image.shape, 4))
        overlay[gt_mask] = [1.0, 0.2, 0.2, 0.25]
        ax.imshow(overlay)

        # Evolution trail (faint)
        n_snap = len(hist)
        for k, (xh, yh) in enumerate(hist[:-1]):
            alpha_val = 0.10 + 0.25 * (k / max(n_snap - 1, 1))
            xc, yc = _close_contour(xh, yh)
            ax.plot(xc, yc, color=col, lw=0.7, alpha=alpha_val)

        # Final contour
        xc, yc = _close_contour(xf, yf)
        ax.plot(xc, yc, color=col, lw=2.5, alpha=1.0, label="Final contour")
        ax.scatter(xf, yf, color=col, s=8, alpha=0.6, zorder=5)

        ax.set_title(title, color="white", fontsize=12, pad=8)
        ax.axis("off")

        # Dice badge
        badge_col = "#00CC66" if col == "#00FF88" else "#FF8C00"
        ax.text(5, image.shape[0] - 8,
                f"DSC = {trad_dice:.4f}" if col == "#FFD700" else f"DSC = {gvf_dice:.4f}",
                color=badge_col, fontsize=11, fontweight="bold",
                va="bottom")

    fig.suptitle("Active Contour Comparison: Traditional Snake vs GVF Snake\n"
                 "Red overlay = Ground Truth Tumour",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – Dice & Metrics summary bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_bar(metrics_dict, save_path=None):
    """
    metrics_dict = {
        'Traditional Snake': {'Dice': 0.xx, 'Precision': 0.xx, 'Recall': 0.xx},
        'GVF Snake':         {'Dice': 0.xx, 'Precision': 0.xx, 'Recall': 0.xx},
    }
    """
    models = list(metrics_dict.keys())
    metric_names = ["Dice", "Precision", "Recall"]
    colours = ["#FF8C00", "#00CC66"]

    x = np.arange(len(metric_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#141414")

    for i, (model, col) in enumerate(zip(models, colours)):
        vals = [metrics_dict[model][m] for m in metric_names]
        bars = ax.bar(x + (i - 0.5) * width, vals, width,
                      label=model, color=col, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", color="white", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, color="white", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title("Segmentation Metrics Comparison", color="white",
                 fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

    legend = ax.legend(facecolor="#1a1a1a", edgecolor="white",
                       labelcolor="white", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 – Animated GIF of GVF snake evolution
# ─────────────────────────────────────────────────────────────────────────────

def animate_snake_evolution(image, history, gt_mask,
                             title="GVF Snake Evolution",
                             colour="#00FF88",
                             save_path=None):
    """Create an animated GIF showing the snake converging."""
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*image.shape, 4))
    overlay[gt_mask] = [1.0, 0.2, 0.2, 0.30]
    ax.imshow(overlay)
    ax.axis("off")

    line, = ax.plot([], [], color=colour, lw=2)
    pts = ax.scatter([], [], color=colour, s=12, zorder=5)
    iteration_text = ax.text(5, 10, "", color="white", fontsize=10)

    def init():
        line.set_data([], [])
        return line, pts, iteration_text

    def update(frame):
        x, y = history[frame]
        xc, yc = _close_contour(x, y)
        line.set_data(xc, yc)
        pts.set_offsets(np.column_stack([x, y]))
        iteration_text.set_text(f"Iteration {frame * 10}")
        return line, pts, iteration_text

    ax.set_title(title, color="white", fontsize=12)

    anim = FuncAnimation(fig, update, frames=len(history),
                         init_func=init, blit=True, interval=80)

    if save_path and save_path.endswith(".gif"):
        writer = PillowWriter(fps=12)
        anim.save(save_path, writer=writer)
        print(f"  Saved: {save_path}")

    plt.close(fig)
    return anim


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 – Energy convergence plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_convergence(trad_energies, gvf_energies, save_path=None):
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#0d0d0d")
    ax.set_facecolor("#141414")

    iters = np.arange(len(trad_energies))
    ax.plot(iters, trad_energies, color="#FF8C00", lw=2,
            label="Traditional Snake")
    ax.plot(np.arange(len(gvf_energies)), gvf_energies, color="#00CC66",
            lw=2, label="GVF Snake")

    ax.set_xlabel("Iteration", color="white", fontsize=12)
    ax.set_ylabel("Contour Energy (normalised)", color="white", fontsize=12)
    ax.set_title("Energy Convergence During Snake Evolution",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.yaxis.grid(True, color="#333", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(facecolor="#1a1a1a", edgecolor="white",
              labelcolor="white", fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    return fig
