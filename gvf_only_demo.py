"""gvf_only_demo.py — Show GVF snake only, highest accuracy"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_mri     import generate_brain_mri
from snake_traditional import init_circle
from gvf               import run_gvf_snake
from metrics           import contour_to_mask, dice_coefficient, precision_recall
from snake_art         import draw_snake_on_axes, animate_snake

def run_gvf_demo(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    image, gt_mask, (tcy,tcx), (try_,trx) = generate_brain_mri(size=256)
    avg_r = (try_+trx)/2

    print("Running GVF Snake (high accuracy mode)...")
    x0,y0 = init_circle((tcx,tcy), avg_r+2, n_points=200)
    gx,gy,ghist,u,v,emap = run_gvf_snake(
        image, x0.copy(), y0.copy(),
        alpha=0.005, beta=1.0, gamma=1.5,
        mu=0.25, sigma=1.5, gvf_iter=500,
        snake_iter=5000, dt=0.12, store_every=80,
        force_scale=6.0, intensity_weight=0.3,
        max_radius_factor=1.06)

    gvf_mask = contour_to_mask(gx, gy, image.shape)
    dice = dice_coefficient(gvf_mask, gt_mask)
    prec, rec = precision_recall(gvf_mask, gt_mask)
    print(f"  Dice={dice:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    # ── Single GVF result figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14,6), facecolor="#080810")
    overlay = np.zeros((*image.shape,4))
    overlay[gt_mask] = [1,0.2,0.2,0.30]

    # Left: input MRI
    axes[0].set_facecolor("#080810")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].imshow(overlay, zorder=1)
    axes[0].set_title("Input MRI  (red = Ground Truth Tumour)",
                      color="white", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Right: GVF result
    axes[1].set_facecolor("#080810")
    axes[1].imshow(image, cmap="gray", vmin=0, vmax=1, alpha=0.82)
    axes[1].imshow(overlay, zorder=1)
    draw_snake_on_axes(axes[1], gx, gy, progress=0.95,
                       show_scales=True, show_head=True, lw_base=3.5)
    axes[1].text(5, image.shape[0]-6,
                 f"Dice = {dice:.4f}   Precision = {prec:.4f}   Recall = {rec:.4f}",
                 color="#00FF88", fontsize=11, fontweight="bold", va="bottom",
                 bbox=dict(facecolor="black", alpha=0.55,
                           boxstyle="round,pad=0.3", edgecolor="none"))
    axes[1].set_title("GVF Snake Result — Body exactly on tumour boundary",
                      color="#00FF88", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("GVF Active Contour Segmentation  |  Xu & Prince (1998)",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "gvf_result.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print(f"  Saved: {p}")

    # ── Animation ─────────────────────────────────────────────────────────────
    animate_snake(image, gt_mask, ghist,
                  title="GVF Snake — Converging on Brain Tumour",
                  save_path=os.path.join(output_dir,"gvf_snake_anim.gif"), fps=10)
    return dice

if __name__ == "__main__":
    run_gvf_demo(output_dir="outputs/gvf_only")
