"""
demo.py
-------
Brain Tumour Segmentation — Full Pipeline Demo
"""

import sys, os, argparse
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_mri     import generate_brain_mri
from snake_traditional import (run_traditional_snake, run_matrix_snake,
                                init_circle, compute_external_energy,
                                build_snake_matrix)
from gvf               import run_gvf_snake
from metrics           import contour_to_mask, dice_coefficient, precision_recall
from visualization     import (plot_mri_and_gt, plot_gvf_field, plot_comparison,
                                plot_metrics_bar, animate_snake_evolution,
                                plot_energy_convergence)


def contour_energy(x, y, image, alpha=0.015, beta=0.1, sigma=2.5):
    d1x = np.roll(x,-1)-x; d1y = np.roll(y,-1)-y
    d2x = np.roll(x,-1)-2*x+np.roll(x,1); d2y = np.roll(y,-1)-2*y+np.roll(y,1)
    e_int = alpha*(d1x**2+d1y**2).sum() + beta*(d2x**2+d2y**2).sum()
    edge_map, _, _ = compute_external_energy(image, sigma=sigma)
    h, w = image.shape
    coords = np.array([np.clip(y,0,h-1), np.clip(x,0,w-1)])
    return (e_int - map_coordinates(edge_map, coords, order=1).sum()) / len(x)


def main(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    sep = "=" * 60
    print(f"\n{sep}\n Brain Tumour Segmentation — Active Contour Demo\n{sep}")

    # 1. MRI
    print("\n[1/8] Generating synthetic brain MRI...")
    image, gt_mask, (tcy, tcx), (try_, trx) = generate_brain_mri(size=256)
    avg_r = (try_ + trx) / 2
    print(f"      Centre ({tcx},{tcy})  radii ({trx},{try_})  avg={avg_r:.1f}")
    plot_mri_and_gt(image, gt_mask,
                    save_path=os.path.join(output_dir, "fig1_mri_gt.png"))

    # 2. Show instability (bad parameters → collapse)
    print("\n[2/8] Demonstrating traditional snake INSTABILITY...")
    ux0, uy0 = init_circle(center=(tcx, tcy), radius=avg_r+5, n_points=80)
    ux, uy, unstable_hist = run_matrix_snake(
        image, ux0.copy(), uy0.copy(),
        alpha=0.001, beta=0.001, gamma=4.0,
        sigma=2.0, n_iter=150, store_every=5)
    r_final = np.sqrt(((ux-ux.mean())**2+(uy-uy.mean())**2).mean())
    print(f"      Started r={avg_r+5:.1f}  →  Final r={r_final:.1f} px (collapse)")

    # Save instability figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    for ax in axes: ax.set_facecolor("#0d0d0d")
    panels = [
        (ux0, uy0, "#FFD700", "Initial Contour\n(near tumour)"),
        (*unstable_hist[len(unstable_hist)//2], "#FF8C00", "Mid-Evolution\nShrinking..."),
        (ux, uy, "#FF4444", f"Collapsed\nRadius = {r_final:.1f} px  ← INSTABILITY"),
    ]
    for ax, (xp, yp, col, ttl) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.plot(np.append(xp,xp[0]), np.append(yp,yp[0]), color=col, lw=2.5)
        ax.set_title(ttl, color=col if col=="#FF4444" else "white", fontsize=11)
        ax.axis("off")
    fig.suptitle("Traditional Snake Instability (α=β=0.001) — Checkpoint 1 Finding\n"
                 "Contour collapses to a point without correct parameter tuning",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_instability.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig); print("  Saved: fig2_instability.png")

    # 3. Traditional snake (stable)
    print("\n[3/8] Traditional snake (stable, scikit-image)...")
    tx0, ty0 = init_circle(center=(tcx, tcy), radius=avg_r+8, n_points=100)
    trad_x, trad_y, trad_hist = run_traditional_snake(
        image, tx0.copy(), ty0.copy(),
        alpha=0.015, beta=0.1, gamma=0.01,
        sigma=2.5, w_line=0.0, w_edge=1.0,
        n_iter=3000, store_every=60)
    trad_mask = contour_to_mask(trad_x, trad_y, image.shape)
    trad_dice = dice_coefficient(trad_mask, gt_mask)
    trad_prec, trad_rec = precision_recall(trad_mask, gt_mask)
    print(f"      Dice={trad_dice:.4f}  P={trad_prec:.4f}  R={trad_rec:.4f}")

    # 4. GVF snake (far initialisation)
    print("\n[4/8] GVF snake (initialised FAR from tumour)...")
    # GVF initialized farther — demonstrates large capture range
    gx0, gy0 = init_circle(center=(tcx, tcy), radius=avg_r+22, n_points=100)
    gvf_x, gvf_y, gvf_hist, u, v, edge_map = run_gvf_snake(
        image, gx0.copy(), gy0.copy(),
        alpha=0.015, beta=0.1, gamma=0.01,
        mu=0.25, sigma=2.5, kappa=0,
        gvf_iter=300, snake_iter=5000, dt=0.2, store_every=100)
    gvf_mask = contour_to_mask(gvf_x, gvf_y, image.shape)
    gvf_dice = dice_coefficient(gvf_mask, gt_mask)
    gvf_prec, gvf_rec = precision_recall(gvf_mask, gt_mask)
    print(f"      Dice={gvf_dice:.4f}  P={gvf_prec:.4f}  R={gvf_rec:.4f}")

    # 5. GVF field (artistic)
    print("\n[5/8] GVF field visualisation...")
    plot_gvf_field(image, u, v, edge_map, step=10, scale=20,
                   save_path=os.path.join(output_dir, "fig3_gvf_field.png"))

    # 6. Comparison
    print("\n[6/8] Side-by-side comparison...")
    plot_comparison(image, gt_mask, trad_hist, gvf_hist,
                    (trad_x, trad_y), (gvf_x, gvf_y),
                    trad_dice, gvf_dice,
                    save_path=os.path.join(output_dir, "fig4_comparison.png"))
    plot_metrics_bar(
        {"Traditional Snake": {"Dice": trad_dice, "Precision": trad_prec, "Recall": trad_rec},
         "GVF Snake":         {"Dice": gvf_dice,  "Precision": gvf_prec,  "Recall": gvf_rec}},
        save_path=os.path.join(output_dir, "fig5_metrics.png"))

    # 7. Energy + animations
    print("\n[7/8] Energy convergence...")
    def norm(hist):
        e = np.array([contour_energy(x, y, image) for x, y in hist])
        r = e.max()-e.min(); return (e-e.min())/r if r>0 else e*0
    plot_energy_convergence(norm(trad_hist), norm(gvf_hist),
                             save_path=os.path.join(output_dir, "fig6_energy.png"))

    print("\n[8/8] Animations...")
    animate_snake_evolution(image, gvf_hist, gt_mask,
                             title="GVF Snake Converging on Brain Tumour",
                             colour="#00FF88",
                             save_path=os.path.join(output_dir, "fig7_gvf_anim.gif"))
    animate_snake_evolution(image, trad_hist, gt_mask,
                             title="Traditional Snake (stable)",
                             colour="#FFD700",
                             save_path=os.path.join(output_dir, "fig8_trad_anim.gif"))

    print(f"\n{sep}\n RESULTS SUMMARY\n{sep}")
    print(f"{'Method':<25} {'Dice':>8} {'Precision':>12} {'Recall':>10}")
    print("-"*60)
    print(f"{'Traditional Snake':<25} {trad_dice:>8.4f} {trad_prec:>12.4f} {trad_rec:>10.4f}")
    print(f"{'GVF Snake':<25} {gvf_dice:>8.4f} {gvf_prec:>12.4f} {gvf_rec:>10.4f}")
    print(sep)
    print(f"\nAll figures → {output_dir}/")
    for f in sorted(os.listdir(output_dir)):
        print(f"  • {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="outputs")
    main(ap.parse_args().output_dir)
