"""
run_demo.py  —  ENTRY POINT
============================
USAGE
-----
  python run_demo.py                                         # synthetic MRI only
  python run_demo.py --dataset_path "C:/full/path/to/lgg"  # + real dataset

OUTPUTS
-------
  outputs/synthetic/   figures from synthetic MRI
  outputs/dataset/     figures from real dataset (separate, no overlap)

NOTE: Snakes do NOT train. They are math optimisation — run on each image directly.
"""
import argparse, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_synthetic():
    from comparison_fixed import run_comparison
    out = "outputs/synthetic"
    os.makedirs(out, exist_ok=True)
    print("\n" + "="*55)
    print("  SYNTHETIC MRI DEMO  (no dataset needed)")
    print("="*55)
    run_comparison(output_dir=out)
    print(f"\n  Saved to: {out}/")
    for f in sorted(os.listdir(out)):
        print(f"    - {f}")


def run_dataset(path):
    out = "outputs/dataset"
    os.makedirs(out, exist_ok=True)

    if not os.path.exists(path):
        print(f"\nERROR: Path not found: {path}")
        print("Use the actual folder path where you downloaded the dataset.")
        print("Example:  C:/Users/yourname/.cache/kagglehub/datasets/.../lgg-mri-segmentation")
        return

    masks = glob.glob(os.path.join(path, "**", "*_mask.*"), recursive=True)
    if not masks:
        print(f"\nERROR: No *_mask.* files found inside: {path}")
        print("Make sure the path points to the folder containing patient subfolders")
        print("like: TCGA_CS_4941_19960909/")
        return

    print(f"\n  Found {len(masks)} mask files. Running evaluation on 10 slices...")
    from evaluate_dataset import evaluate
    evaluate(dataset_path=path, test_ratio=0.20, n_eval=10, output_dir=out)
    print(f"\n  Saved to: {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default=None,
                    help="Full path to LGG MRI dataset folder")
    args = ap.parse_args()

    run_synthetic()

    if args.dataset_path:
        run_dataset(args.dataset_path)
    else:
        print("\nTip: Add --dataset_path to also evaluate on real MRI slices.")
