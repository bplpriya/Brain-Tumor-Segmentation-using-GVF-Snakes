"""snake_art.py — Circle head, closed body, stays on tumor boundary"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d


def _get_cmap(progress):
    colors = (["#00FFCC","#00CC88","#009966"] if progress < 0.5
              else ["#00FF44","#44CC00","#226600"])
    return LinearSegmentedColormap.from_list("s", colors)

def _smooth(x, y, sigma=1.2):
    return (gaussian_filter1d(x, sigma, mode='wrap'),
            gaussian_filter1d(y, sigma, mode='wrap'))


def draw_snake_on_axes(ax, x, y, progress=0.0, lw_base=3.0,
                       show_scales=True, show_head=True, label=None):
    n  = len(x)
    xs, ys = _smooth(x, y)
    cmap   = _get_cmap(progress)

    # ── CLOSED BODY ───────────────────────────────────────────────────────────
    pts   = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    close = np.array([[[xs[-1], ys[-1]], [xs[0], ys[0]]]])
    segs  = np.concatenate([segs, close], axis=0)
    t     = np.linspace(0, 1, len(segs))

    ax.add_collection(LineCollection(segs, linewidths=lw_base+3,
                                     colors=(0,0,0,0.35), zorder=3))
    ax.add_collection(LineCollection(segs, array=t, cmap=cmap,
                                     linewidths=lw_base, zorder=4, alpha=0.95))
    hl = LinearSegmentedColormap.from_list("h",["#FFFFFF55","#AAFFCC22"])
    ax.add_collection(LineCollection(segs, array=t, cmap=hl,
                                     linewidths=lw_base*0.28, zorder=5, alpha=0.6))

    # ── SCALES ────────────────────────────────────────────────────────────────
    if show_scales and n > 20:
        sc = cmap(0.7)
        for i in np.arange(3, n-2, max(1, n//24)).astype(int):
            xi, yi = xs[i], ys[i]
            tx = xs[(i+1)%n]-xs[(i-1)%n]
            ty = ys[(i+1)%n]-ys[(i-1)%n]
            tl = np.hypot(tx,ty)+1e-10; tx/=tl; ty/=tl
            nx,ny = -ty,tx; s = lw_base*0.85
            verts = np.array([
                [xi+ny*s,     yi-nx*s],
                [xi+tx*s*.5,  yi+ty*s*.5],
                [xi-ny*s*.6,  yi+nx*s*.6],
                [xi-tx*s*.5,  yi-ty*s*.5]])
            ax.add_patch(Polygon(verts, closed=True, facecolor=sc,
                                 edgecolor='black', lw=0.4, alpha=0.65, zorder=6))

    # ── HEAD: small circle ON the boundary + eyes + tangent tongue ────────────
    if show_head and n > 5:
        hx, hy = float(x[0]), float(y[0])   # raw point on boundary

        # Outward direction
        ox = hx - xs.mean();  oy = hy - ys.mean()
        ol = np.hypot(ox,oy)+1e-10; ox/=ol; oy/=ol

        # Tangent direction
        fwd = min(5, n-1)
        tx2 = float(x[fwd]-x[-fwd]); ty2 = float(y[fwd]-y[-fwd])
        tl2 = np.hypot(tx2,ty2)+1e-10; tx2/=tl2; ty2/=tl2

        # Small circle — same size as body line width, sits ON the oval
        hr = lw_base * 1.3
        head_col = _get_cmap(0.1)(0.15)
        ax.add_patch(plt.Circle((hx, hy), hr,
                                facecolor=head_col, edgecolor='black',
                                linewidth=0.7, alpha=0.96, zorder=7))

        # Eyes: on the outer half of the circle
        for sign in (+1, -1):
            ex = hx + ox*hr*0.35 + tx2*hr*0.42*sign
            ey = hy + oy*hr*0.35 + ty2*hr*0.42*sign
            ax.plot(ex, ey, 'o', color='white',
                    ms=lw_base*0.72, mec='#222', mew=0.4, zorder=9)
            ax.plot(ex+ox*0.5, ey+oy*0.5,
                    'o', color='black', ms=lw_base*0.28, zorder=10)

        # Tongue: along tangent (stays on oval, doesn't go outside tumor)
        tip_x = hx + tx2*hr;  tip_y = hy + ty2*hr
        fork = hr*0.55
        for sign in (+1,-1):
            ax.plot([tip_x, tip_x+tx2*hr*0.9+sign*ox*fork*0.35],
                    [tip_y, tip_y+ty2*hr*0.9+sign*oy*fork*0.35],
                    color='#FF2255', lw=0.85, solid_capstyle='round', zorder=8)

    if label:
        ax.text(xs.mean(), ys.mean()-18, label, color='white', fontsize=9,
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black',
                          alpha=0.6, edgecolor='none'), zorder=11)


def animate_snake(image, gt_mask, history, title="Snake Evolution",
                  save_path=None, fps=10):
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.patches as mpatches
    n_frames = len(history)
    fig, ax = plt.subplots(figsize=(6,6), facecolor="#050510")
    ax.set_facecolor("#050510")
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, alpha=0.85)
    if gt_mask is not None:
        ov = np.zeros((*image.shape,4)); ov[gt_mask]=[1,0.2,0.2,0.30]
        ax.imshow(ov, zorder=1)
        ax.legend(handles=[mpatches.Patch(color=(1,0.2,0.2,0.5),
                  label="Ground Truth")], loc="lower left", fontsize=8,
                  facecolor="#111", edgecolor="#555", labelcolor="white")
    ax.axis("off")
    ax.set_title(title, color="white", fontsize=11, pad=6, fontweight="bold")
    txt = ax.text(5,14,"",color="#00FF88",fontsize=10,fontweight="bold",zorder=12)
    _art=[]
    def _clear():
        for a in _art:
            try: a.remove()
            except: pass
        _art.clear()
    def update(frame):
        _clear()
        pp=set(ax.patches); pl=set(ax.lines); pc=set(ax.collections)
        draw_snake_on_axes(ax, *history[frame],
                           progress=frame/max(n_frames-1,1),
                           lw_base=2.8, show_scales=True, show_head=True)
        _art.extend([p for p in ax.patches    if p not in pp])
        _art.extend([l for l in ax.lines      if l not in pl])
        _art.extend([c for c in ax.collections if c not in pc])
        txt.set_text(f"Iteration {frame*10}")
        return _art
    anim = FuncAnimation(fig, update, frames=n_frames,
                         init_func=lambda:[], interval=80, blit=False)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print(f"  Saved: {save_path}")
    plt.close(fig)
    return anim
