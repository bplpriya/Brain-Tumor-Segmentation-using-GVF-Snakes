"""
snake_traditional.py
--------------------
Parametric Active Contour Model (Traditional Snake)
Based on: Kass, Witkin & Terzopoulos (1988)

Mathematical foundation
-----------------------
Total snake energy:
    E_snake = ∫ [α|v'(s)|² + β|v''(s)|²] ds  +  ∫ E_ext(v(s)) ds

Discretised Euler implicit update:
    (A + γI) · x^{t+1} = γ · x^t + f_x(x^t, y^t)
    (A + γI) · y^{t+1} = γ · y^t + f_y(x^t, y^t)

where A is pentadiagonal matrix built from α (elasticity) and β (stiffness).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, map_coordinates


def build_snake_matrix(n: int, alpha: float, beta: float) -> np.ndarray:
    """
    Build n×n pentadiagonal internal-energy matrix A (periodic).

    Band  0 :  2α + 6β
    Band ±1 : -α − 4β
    Band ±2 :  β
    """
    row = np.zeros(n)
    row[0]  =  2 * alpha + 6 * beta
    row[1]  = -alpha - 4 * beta
    row[2]  =  beta
    row[-1] = -alpha - 4 * beta
    row[-2] =  beta
    A = np.zeros((n, n))
    for i in range(n):
        A[i] = np.roll(row, i)
    return A


def compute_external_energy(image: np.ndarray, sigma: float = 2.0):
    """
    External potential combining edge + line energies.

    Returns (edge_map, fx, fy) — forces scaled to [-1, 1].
    """
    smoothed = gaussian_filter(image.astype(np.float64), sigma=sigma)
    gy_s = sobel(smoothed, axis=0)
    gx_s = sobel(smoothed, axis=1)
    edge_map = gx_s ** 2 + gy_s ** 2

    fx_edge = sobel(edge_map, axis=1)
    fy_edge = sobel(edge_map, axis=0)
    fx_line = sobel(smoothed, axis=1)
    fy_line = sobel(smoothed, axis=0)

    w_edge, w_line = 1.0, 0.5
    fx = w_edge * fx_edge + w_line * fx_line
    fy = w_edge * fy_edge + w_line * fy_line

    scale = np.sqrt(fx ** 2 + fy ** 2).max() + 1e-10
    return edge_map, fx / scale, fy / scale


def init_circle(center, radius, n_points=80):
    """Return (x, y) arrays for a circular snake initialisation."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cx, cy = center
    return (cx + radius * np.cos(t)).astype(np.float64), \
           (cy + radius * np.sin(t)).astype(np.float64)


def run_traditional_snake(
    image: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    alpha: float = 0.015,
    beta:  float = 0.1,
    gamma: float = 0.01,
    sigma: float = 2.5,
    n_iter: int  = 2500,
    store_every: int = 50,
    w_line: float = 0.0,
    w_edge: float = 1.0,
):
    """
    Traditional snake using scikit-image's stable implementation
    with the Kass et al. parameter interface (alpha, beta, gamma).

    Returns
    -------
    x, y     : final contour (col, row)
    history  : list of (x, y) snapshots
    """
    from skimage.segmentation import active_contour

    snake = np.column_stack([y0, x0])   # (N,2) in (row, col)
    history = [(x0.copy(), y0.copy())]
    chunk = store_every

    for _ in range(n_iter // chunk):
        snake = active_contour(
            gaussian_filter(image, sigma=sigma),
            snake,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            w_line=w_line,
            w_edge=w_edge,
            max_num_iter=chunk,
            boundary_condition='periodic',
        )
        history.append((snake[:, 1].copy(), snake[:, 0].copy()))

    return snake[:, 1], snake[:, 0], history


def run_matrix_snake(
    image: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    alpha: float = 0.01,
    beta:  float = 0.10,
    gamma: float = 4.0,
    sigma: float = 2.0,
    n_iter: int  = 300,
    store_every: int = 10,
):
    """
    Pure pentadiagonal matrix snake — demonstrates the numerical
    instability (shrinking to a point) described in Checkpoint 1.
    """
    n = len(x0)
    A = build_snake_matrix(n, alpha, beta)
    inv_matrix = np.linalg.inv(A + gamma * np.eye(n))
    _, fx, fy = compute_external_energy(image, sigma=sigma)

    x, y = x0.copy(), y0.copy()
    history = [(x.copy(), y.copy())]
    h, w = image.shape

    for i in range(n_iter):
        coords = np.array([np.clip(y, 0, h-1), np.clip(x, 0, w-1)])
        force_x = map_coordinates(fx, coords, order=1)
        force_y = map_coordinates(fy, coords, order=1)
        x = inv_matrix @ (gamma * x + force_x)
        y = inv_matrix @ (gamma * y + force_y)
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)
        if (i + 1) % store_every == 0:
            history.append((x.copy(), y.copy()))

    return x, y, history
