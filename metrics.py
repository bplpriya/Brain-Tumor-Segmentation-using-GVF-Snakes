"""
metrics.py
----------
Evaluation utilities for brain tumour segmentation.
"""

import numpy as np
from matplotlib.path import Path


def contour_to_mask(x, y, shape):
    """
    Convert a closed contour (x, y) to a binary mask using ray-casting.

    Parameters
    ----------
    x, y  : contour column / row coordinates (N,)
    shape : (H, W) of the output mask

    Returns
    -------
    mask : bool array of shape (H, W)
    """
    h, w = shape
    # Build a closed polygon path
    verts = np.stack([x, y], axis=1)            # (N, 2) in (col, row) = (x, y)
    path = Path(verts)

    # Create grid of pixel centres
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack([cols.ravel(), rows.ravel()], axis=1)

    mask = path.contains_points(points).reshape(h, w)
    return mask


def dice_coefficient(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Compute the Sørensen–Dice Similarity Coefficient.

        DSC = 2 |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask_pred : predicted binary mask
    mask_gt   : ground-truth binary mask

    Returns
    -------
    dice : float in [0, 1]   (1 = perfect overlap)
    """
    pred = mask_pred.astype(bool)
    gt   = mask_gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()

    if total == 0:
        return 1.0  # both empty → trivially identical

    return 2.0 * intersection / total


def hausdorff_distance(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """
    Compute the symmetric Hausdorff distance between two binary masks
    (boundary pixels only).
    """
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    def boundary(m):
        return m & ~binary_erosion(m)

    b_pred = np.argwhere(boundary(mask_pred.astype(bool)))
    b_gt   = np.argwhere(boundary(mask_gt.astype(bool)))

    if len(b_pred) == 0 or len(b_gt) == 0:
        return float("inf")

    tree_pred = cKDTree(b_pred)
    tree_gt   = cKDTree(b_gt)

    d_pred = tree_pred.query(b_gt)[0].max()
    d_gt   = tree_gt.query(b_pred)[0].max()

    return max(d_pred, d_gt)


def precision_recall(mask_pred: np.ndarray, mask_gt: np.ndarray):
    """Return (precision, recall) for binary segmentation masks."""
    pred = mask_pred.astype(bool)
    gt   = mask_gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
