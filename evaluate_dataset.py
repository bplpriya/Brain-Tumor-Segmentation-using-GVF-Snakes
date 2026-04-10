"""
evaluate_dataset.py
-------------------
Batch evaluation of GVF snake on the LGG MRI Segmentation Dataset.
Splits by patient into train/test, runs GVF on test set, reports Dice.

Usage:
    import kagglehub
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

    python evaluate_dataset.py --dataset_path <path> --n_test 20
"""

import sys, os, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset         import LGGDataset
from snake_traditional import init_circle
from gvf             import run_gvf_snake
from snake_traditional import run_traditional_snake
from metrics         import contour_to_mask, dice_coefficient


def auto_center(image):
    """Detect brightest cluster as tumour centre."""
    from scipy.ndimage import label, center_of_mass
    thresh = image > (image.mean() + 1.2 * image.std())
    labeled, n = label(thresh)
    if n == 0:
        h, w = image.shape
        return (h//2, w//2), 25
    sizes = [(labeled == i).sum() for i in range(1, n+1)]
    biggest = np.argmax(sizes) + 1
    region  = labeled == biggest
    cy, cx  = center_of_mass(region)
    ys, xs  = np.where(region)
    r = max(15, np.sqrt(((ys-cy)**2+(xs-cx)**2).mean()))
    return (int(cy), int(cx)), int(r * 1.4)


def segment_one(image, alpha=0.015, beta=0.1, mu=0.25, sigma=2.5,
                gvf_iter=200, snake_iter=3000, init_extra=25, use_gvf=True):
    """Run one GVF or traditional snake on a single image. Returns final contour."""
    (cy, cx), radius = auto_center(image)
    init_r = radius + init_extra
    x0, y0 = init_circle((cx, cy), init_r, n_points=100)

    if use_gvf:
        xf, yf, _, _, _, _ = run_gvf_snake(
            image, x0, y0,
            alpha=alpha, beta=beta, gamma=0.01,
            mu=mu, sigma=sigma, kappa=0,
            gvf_iter=gvf_iter, snake_iter=snake_iter,
            dt=0.2, store_every=snake_iter,  # only keep final
        )
    else:
        xf, yf, _ = run_traditional_snake(
            image, x0, y0,
            alpha=alpha, beta=beta, gamma=0.01,
            sigma=sigma, w_line=0.0, w_edge=1.0,
            n_iter=snake_iter, store_every=snake_iter,
        )
    return xf, yf


def evaluate(dataset_path: str,
             test_ratio:    float = 0.20,
             n_eval:        int   = None,
             alpha:         float = 0.015,
             mu:            float = 0.25,
             output_dir:    str   = "eval_outputs"):

    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print(" LGG MRI Dataset Evaluation")
    print("=" * 60)

    # Load & split
    ds = LGGDataset(dataset_path, size=256, only_tumor=True)
    train_ds, test_ds = ds.split(test_ratio=test_ratio)

    if n_eval:
        test_ds.samples = test_ds.samples[:n_eval]
        print(f"  (Evaluating first {n_eval} test slices)")

    print(f"\n  Evaluating {len(test_ds)} test slices...")
    print(f"  Parameters: α={alpha}, μ={mu}\n")

    gvf_dices   = []
    trad_dices  = []
    timings     = []
    failures    = []

    for i in range(len(test_ds)):
        image, gt_mask, path = test_ds[i]
        name = os.path.basename(path)
        t0 = time.time()

        try:
            # GVF
            xg, yg = segment_one(image, alpha=alpha, mu=mu,
                                  gvf_iter=150, snake_iter=2000,
                                  init_extra=20, use_gvf=True)
            gvf_mask = contour_to_mask(xg, yg, image.shape)
            gd = dice_coefficient(gvf_mask, gt_mask)

            # Traditional
            xt, yt = segment_one(image, alpha=alpha, mu=mu,
                                  gvf_iter=150, snake_iter=2000,
                                  init_extra=20, use_gvf=False)
            trad_mask = contour_to_mask(xt, yt, image.shape)
            td = dice_coefficient(trad_mask, gt_mask)

            gvf_dices.append(gd)
            trad_dices.append(td)
            timings.append(time.time() - t0)

            flag = "★" if gd > td else " "
            print(f"  [{i+1:3d}/{len(test_ds)}] {flag} "
                  f"GVF={gd:.4f}  Trad={td:.4f}  "
                  f"Δ={gd-td:+.4f}  {name[:30]}")

        except Exception as ex:
            failures.append((name, str(ex)))
            print(f"  [{i+1:3d}/{len(test_ds)}] ⚠  FAILED: {name} — {ex}")

    if not gvf_dices:
        print("No results.")
        return

    # Summary
    print("\n" + "="*60)
    print(f"{'Metric':<30} {'GVF':>8} {'Traditional':>14}")
    print("-"*60)
    print(f"{'Mean Dice':<30} {np.mean(gvf_dices):>8.4f} {np.mean(trad_dices):>14.4f}")
    print(f"{'Median Dice':<30} {np.median(gvf_dices):>8.4f} {np.median(trad_dices):>14.4f}")
    print(f"{'Std Dice':<30} {np.std(gvf_dices):>8.4f} {np.std(trad_dices):>14.4f}")
    print(f"{'Min Dice':<30} {np.min(gvf_dices):>8.4f} {np.min(trad_dices):>14.4f}")
    print(f"{'Max Dice':<30} {np.max(gvf_dices):>8.4f} {np.max(trad_dices):>14.4f}")
    print(f"{'GVF wins (#slices)':<30} {(np.array(gvf_dices)>np.array(trad_dices)).sum():>8d} "
          f"/ {len(gvf_dices)}")
    print(f"{'Mean time/slice (s)':<30} {np.mean(timings):>8.2f}")
    print("="*60)

    if failures:
        print(f"\n  {len(failures)} slices failed.")

    # Plot
    _plot_results(gvf_dices, trad_dices, output_dir)
    _save_csv(gvf_dices, trad_dices, output_dir)

    return gvf_dices, trad_dices


def _plot_results(gvf_dices, trad_dices, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#141414")

    n = len(gvf_dices)
    x = np.arange(n)

    # 1. Per-slice Dice
    axes[0].plot(x, gvf_dices,  color="#00CC66", lw=1.2, label="GVF", alpha=0.8)
    axes[0].plot(x, trad_dices, color="#FF6600", lw=1.2, label="Trad", alpha=0.8)
    axes[0].axhline(np.mean(gvf_dices),  color="#00FF88", lw=1.5, ls="--", alpha=0.7)
    axes[0].axhline(np.mean(trad_dices), color="#FF8844", lw=1.5, ls="--", alpha=0.7)
    axes[0].set_title("Per-slice Dice", color="white", fontsize=11)
    axes[0].set_xlabel("Test slice #", color="white")
    axes[0].set_ylabel("Dice", color="white")
    axes[0].tick_params(colors="white")
    axes[0].legend(facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
    axes[0].yaxis.grid(True, color="#333", linestyle="--", lw=0.7)

    # 2. Boxplot
    bp = axes[1].boxplot([gvf_dices, trad_dices], patch_artist=True,
                          labels=["GVF", "Traditional"],
                          medianprops=dict(color="white", lw=2))
    for patch, col in zip(bp["boxes"], ["#00CC66", "#FF6600"]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    axes[1].set_title("Dice Distribution", color="white", fontsize=11)
    axes[1].set_ylabel("Dice", color="white")
    axes[1].tick_params(colors="white")
    axes[1].yaxis.grid(True, color="#333", linestyle="--", lw=0.7)
    axes[1].set_facecolor("#141414")

    # 3. Δ Dice (GVF - Trad) per slice
    delta = np.array(gvf_dices) - np.array(trad_dices)
    colors = ["#00CC66" if d >= 0 else "#FF4444" for d in delta]
    axes[2].bar(x, delta, color=colors, alpha=0.8, width=0.8)
    axes[2].axhline(0, color="white", lw=0.8)
    axes[2].axhline(delta.mean(), color="#FFD700", lw=1.5, ls="--",
                    label=f"Mean Δ={delta.mean():+.4f}")
    axes[2].set_title("GVF − Traditional Dice (Δ)", color="white", fontsize=11)
    axes[2].set_xlabel("Test slice #", color="white")
    axes[2].tick_params(colors="white")
    axes[2].legend(facecolor="#1a1a1a", edgecolor="white", labelcolor="white")
    axes[2].yaxis.grid(True, color="#333", linestyle="--", lw=0.7)

    for ax in axes:
        for sp in ax.spines.values(): sp.set_color("#444")

    fig.suptitle("LGG MRI Dataset Evaluation — GVF vs Traditional Snake",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(output_dir, "dataset_evaluation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved: {p}")


def _save_csv(gvf_dices, trad_dices, output_dir):
    p = os.path.join(output_dir, "dice_results.csv")
    with open(p, "w") as f:
        f.write("slice,gvf_dice,trad_dice,delta\n")
        for i, (g, t) in enumerate(zip(gvf_dices, trad_dices)):
            f.write(f"{i},{g:.6f},{t:.6f},{g-t:+.6f}\n")
    print(f"  CSV saved:  {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--test_ratio",   type=float, default=0.20)
    ap.add_argument("--n_eval",       type=int,   default=None,
                    help="Limit to first N test slices (fast test)")
    ap.add_argument("--alpha",        type=float, default=0.015)
    ap.add_argument("--mu",           type=float, default=0.25)
    ap.add_argument("--output_dir",   default="eval_outputs")
    args = ap.parse_args()

    evaluate(dataset_path=args.dataset_path,
             test_ratio=args.test_ratio,
             n_eval=args.n_eval,
             alpha=args.alpha,
             mu=args.mu,
             output_dir=args.output_dir)
