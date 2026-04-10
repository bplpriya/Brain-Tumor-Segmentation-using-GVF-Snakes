"""
synthetic_mri.py
----------------
Generates a synthetic brain MRI slice with a simulated tumor region.
Used for demonstration when no real MRI dataset is available.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_brain_mri(size=256, noise_level=0.04, seed=42):
    """
    Generate a synthetic T1-weighted brain MRI slice with a tumor.

    Returns
    -------
    image      : ndarray (size x size), float64 in [0, 1]  – the MRI image
    ground_truth: ndarray (size x size), bool              – tumor mask (ground truth)
    tumor_center: (row, col) tuple
    tumor_radii : (ry, rx) tuple (semi-axes of the tumor ellipse)
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    cy, cx = size // 2, size // 2

    # ── skull ring ────────────────────────────────────────────────────────────
    skull_outer = 0.90
    skull_inner = 0.82
    Y, X = np.ogrid[:size, :size]
    R = np.sqrt(((Y - cy) / (size * skull_outer / 2)) ** 2 +
                ((X - cx) / (size * skull_outer / 2)) ** 2)
    skull_mask = (R > skull_inner) & (R < skull_outer)
    img[skull_mask] = 0.85

    # ── white matter ─────────────────────────────────────────────────────────
    wm_ry, wm_rx = int(size * 0.36), int(size * 0.38)
    wm = ((Y - cy) / wm_ry) ** 2 + ((X - cx) / wm_rx) ** 2 <= 1
    img[wm] = 0.78

    # ── gray matter cortex ───────────────────────────────────────────────────
    gm_ry, gm_rx = int(size * 0.40), int(size * 0.42)
    gm = ((Y - cy) / gm_ry) ** 2 + ((X - cx) / gm_rx) ** 2 <= 1
    img[gm & ~wm] = 0.55

    # ── ventricles ────────────────────────────────────────────────────────────
    for (vy, vx, vry, vrx) in [
        (cy - 25, cx - 15, 18, 10),
        (cy - 25, cx + 15, 18, 10),
    ]:
        v_mask = ((Y - vy) / vry) ** 2 + ((X - vx) / vrx) ** 2 <= 1
        img[v_mask] = 0.10

    # ── bright tumor ─────────────────────────────────────────────────────────
    tumor_cy, tumor_cx = cy + 35, cx + 40
    tumor_ry, tumor_rx = 28, 22
    tumor = ((Y - tumor_cy) / tumor_ry) ** 2 + ((X - tumor_cx) / tumor_rx) ** 2 <= 1
    img[tumor] = 0.95

    # tumor necrotic core (darker centre)
    core_ry, core_rx = int(tumor_ry * 0.45), int(tumor_rx * 0.45)
    core = ((Y - tumor_cy) / core_ry) ** 2 + ((X - tumor_cx) / core_rx) ** 2 <= 1
    img[core] = 0.60

    # oedema (slightly brighter halo around tumor)
    oedema_ry, oedema_rx = int(tumor_ry * 1.45), int(tumor_rx * 1.45)
    oedema = ((Y - tumor_cy) / oedema_ry) ** 2 + ((X - tumor_cx) / oedema_rx) ** 2 <= 1
    img[oedema & ~tumor] = 0.68

    # ── smooth + noise ────────────────────────────────────────────────────────
    img = gaussian_filter(img, sigma=1.5)
    img += rng.normal(0, noise_level, img.shape)
    img = np.clip(img, 0, 1)

    # ── ground truth mask (whole tumor region) ───────────────────────────────
    gt = ((Y - tumor_cy) / tumor_ry) ** 2 + ((X - tumor_cx) / tumor_rx) ** 2 <= 1

    return img, gt, (tumor_cy, tumor_cx), (tumor_ry, tumor_rx)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img, gt, center, radii = generate_brain_mri()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Synthetic Brain MRI")
    axes[0].axis("off")
    axes[1].imshow(gt, cmap="hot")
    axes[1].set_title("Ground Truth Tumor Mask")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("synthetic_mri_preview.png", dpi=150)
    print("Saved synthetic_mri_preview.png")
