"""
gvf.py  —  Gradient Vector Flow Snake (Xu & Prince 1998)
---------------------------------------------------------
Final stable version. Key fixes:
  1. beta=2.0        — high stiffness, prevents loops
  2. reparametrize   — uniform point spacing every step
  3. 400 GVF iters   — forces diffuse further
  4. skimage sobel   — cleaner edge map
  5. gamma=1.5, dt=0.15, force_scale=5.0  — no overshoot
  6. clip every step — hard image boundary
  7. max_radius*1.05 — cannot leak to brain boundary
  8. image intensity — pulls snake to bright tumor boundary
"""
import numpy as np
from scipy.ndimage import gaussian_filter, laplace, map_coordinates
from skimage.filters import sobel as skimage_sobel


def reparametrize(x, y):
    """Uniform point redistribution — prevents clustering & self-intersection."""
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    d = np.insert(np.cumsum(d), 0, 0)
    if d[-1] < 1e-10:
        return x, y
    d /= d[-1]
    t = np.linspace(0, 1, len(x))
    return np.interp(t, d, x), np.interp(t, d, y)


def compute_gvf(image, mu=0.25, sigma=2.0, n_iter=400, dt=0.15):
    from scipy.ndimage import sobel as sp_sobel
    sm = gaussian_filter(image.astype(np.float64), sigma=sigma)
    edge_map = skimage_sobel(sm)
    emax = edge_map.max()
    if emax > 0:
        edge_map /= emax
    fx = sp_sobel(edge_map, axis=1)
    fy = sp_sobel(edge_map, axis=0)
    b  = fx**2 + fy**2
    u, v = fx.copy(), fy.copy()
    for _ in range(n_iter):
        u += dt * (mu * laplace(u) - b*(u - fx))
        v += dt * (mu * laplace(v) - b*(v - fy))
    sc = np.sqrt(u**2 + v**2).max() + 1e-10
    return u/sc, v/sc, edge_map, fx, fy


def run_gvf_snake(
    image, x0, y0,
    alpha=0.01,
    beta=1.0,             # high — prevents loops and boundary crossing
    gamma=1.5,
    mu=0.25, sigma=2.0,
    kappa=0.0,
    gvf_iter=500, snake_iter=5000,
    dt=0.12, store_every=60,
    force_scale=6.0,
    intensity_weight=0.3, # pulls snake toward bright tumor boundary
    max_radius_factor=1.06,
):
    from snake_traditional import build_snake_matrix

    print("  Computing GVF field (diffusion)...")
    u, v, edge_map, _, _ = compute_gvf(image, mu=mu, sigma=sigma,
                                        n_iter=gvf_iter, dt=dt)

    # Smooth image for intensity force
    img_smooth = gaussian_filter(image.astype(np.float64), sigma=1.5)

    n    = len(x0)
    A    = build_snake_matrix(n, alpha, beta)
    invM = np.linalg.inv(A + gamma * np.eye(n))

    x, y   = x0.copy(), y0.copy()
    cx0    = x.mean();  cy0 = y.mean()
    init_r = np.sqrt(((x-cx0)**2 + (y-cy0)**2).mean())
    max_r  = init_r * max_radius_factor

    history = [(x.copy(), y.copy())]
    h, w = image.shape

    for i in range(snake_iter):
        # Uniform spacing
        x, y = reparametrize(x, y)

        coords = np.array([np.clip(y,0,h-1), np.clip(x,0,w-1)])

        # GVF forces (edge attraction)
        fx_s = map_coordinates(u, coords, order=1) * force_scale
        fy_s = map_coordinates(v, coords, order=1) * force_scale

        # Intensity force — attracts snake toward bright tumor pixels
        img_v = map_coordinates(img_smooth, coords, order=1)
        fx_s += img_v * intensity_weight
        fy_s += img_v * intensity_weight

        # Implicit Euler
        x = invM @ (gamma * x + fx_s)
        y = invM @ (gamma * y + fy_s)

        # Hard image boundary
        x = np.clip(x, 0, w-1)
        y = np.clip(y, 0, h-1)

        # Max-radius: cannot leak to brain/skull boundary
        dists = np.sqrt((x-cx0)**2 + (y-cy0)**2)
        too_far = dists > max_r
        if too_far.any():
            sc = np.where(too_far, max_r/(dists+1e-10), 1.0)
            x = cx0 + (x-cx0)*sc
            y = cy0 + (y-cy0)*sc

        if (i+1) % store_every == 0:
            history.append((x.copy(), y.copy()))

    return x, y, history, u, v, edge_map
