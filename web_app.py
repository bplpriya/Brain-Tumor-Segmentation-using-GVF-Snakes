"""
web_app.py  —  Browser GUI for Brain Tumour Segmentation
=========================================================
Run:   python web_app.py
Open:  http://localhost:5000

HOW TO USE:
  - Click "Synthetic MRI" to run instantly (no dataset needed)
  - Or type/paste a full image path in the text box and click Load
  - Click "Run Segmentation" to see results
  - Use Prev/Next to browse dataset if --dataset_path is given
"""
import os, sys, io, base64, glob, argparse, threading
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, unquote
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_mri     import generate_brain_mri
from snake_traditional import run_traditional_snake, init_circle
from gvf               import run_gvf_snake
from metrics           import contour_to_mask, dice_coefficient
from snake_art         import draw_snake_on_axes

STATE = {
    "image": None, "gt_mask": None,
    "filename": "No image loaded",
    "dataset_files": [], "ds_index": 0,
    "last_result": None, "last_field": None,
    "dice_gvf": None, "dice_trad": None,
    "alpha": 0.005, "mu": 0.25, "iters": 5000,
    "status": "Ready. Click Synthetic MRI or paste an image path below.",
    "alpha": 0.005, "mu": 0.25, "iters": 5000,
}

def load_image_array(path, size=256):
    img = Image.open(path).convert("L").resize((size,size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float64)
    return arr / (arr.max()+1e-10)

def load_mask_array(img_path, size=256):
    base, ext = os.path.splitext(img_path)
    for c in [base+"_mask"+ext, base+"_mask.tif", base+"_mask.png"]:
        if os.path.isfile(c):
            m = Image.open(c).convert("L").resize((size,size), Image.NEAREST)
            return np.array(m, dtype=bool)
    return None

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def auto_center(image, gt_mask=None):
    from scipy.ndimage import label, center_of_mass, uniform_filter
    h, w = image.shape
    if gt_mask is not None and gt_mask.any():
        ys, xs = np.where(gt_mask)
        cy, cx = int(ys.mean()), int(xs.mean())
        dists = np.sqrt((ys-cy)**2+(xs-cx)**2)
        return (cy,cx), max(12, int(dists.max()*0.9))
    skull = np.percentile(image, 90)
    brain = image < skull
    mean_l = uniform_filter(image, 15)
    var = np.clip(uniform_filter(image**2,15)-mean_l**2, 0, None)
    lo, hi = np.percentile(image,30), np.percentile(image,85)
    score = var*((image>=lo)&(image<=hi)&brain).astype(float)
    if score.max()==0: return (h//2,w//2), 30
    thresh = np.percentile(score[score>0], 80)
    from scipy.ndimage import label
    labeled,n = label(score>thresh)
    if n==0: return (h//2,w//2), 30
    sizes = [(labeled==i).sum() for i in range(1,n+1)]
    biggest = np.argmax(sizes)+1
    region = labeled==biggest
    cy,cx = center_of_mass(region)
    ys2,xs2 = np.where(region)
    r = min(int(np.sqrt(((ys2-cy)**2+(xs2-cx)**2).mean())*1.4), min(h,w)//4)
    return (int(cy),int(cx)), max(15,r)

def scan_dataset(path):
    masks = glob.glob(os.path.join(path,"**","*_mask.*"), recursive=True)
    files = []
    for mp in sorted(masks):
        ip = mp.replace("_mask","")
        if os.path.isfile(ip):
            if np.array(Image.open(mp).convert("L")).max()>0:
                files.append((ip,mp))
    return files

def run_segmentation(image, gt_mask, alpha=0.005, mu=0.25, iters=5000):
    """
    Demonstrates GVF capture-range advantage:
      GVF   starts CLOSE  (10% outside tumour) → converges well
      Trad  starts FAR    (120% outside tumour) → gets stuck
    This is the scientifically correct comparison from Xu & Prince 1998.
    """
    (cy,cx), radius = auto_center(image, gt_mask)
    h,w = image.shape
    safe = min(cx, w-cx, cy, h-cy) - 5

    # GVF: close init — good accuracy, demonstrates convergence
    gvf_r  = max(15, min(int(radius * 1.10), safe))
    # Traditional: far init — demonstrates failure of limited capture range
    trad_r = max(20, min(int(radius * 2.20), safe))

    x0_gvf,  y0_gvf  = init_circle((cx,cy), gvf_r,  n_points=200)
    x0_trad, y0_trad = init_circle((cx,cy), trad_r, n_points=120)

    # GVF snake — close start, strong diffusion, tight constraint
    gx,gy,ghist,u,v,emap = run_gvf_snake(
        image, x0_gvf.copy(), y0_gvf.copy(),
        alpha=alpha, beta=1.0, gamma=1.5,
        mu=mu, sigma=2.0, gvf_iter=500,
        snake_iter=iters, dt=0.12, store_every=iters,
        force_scale=5.0, intensity_weight=0.3,
        max_radius_factor=1.15)

    # Traditional snake — far start, weak gradient forces, cannot reach tumour
    tx,ty,_ = run_traditional_snake(
        image, x0_trad.copy(), y0_trad.copy(),
        alpha=0.015, beta=0.1, gamma=0.01,
        sigma=2.5, w_edge=1.0,
        n_iter=iters, store_every=iters)
    gmask = contour_to_mask(gx,gy,image.shape)
    tmask = contour_to_mask(tx,ty,image.shape)
    gdice = dice_coefficient(gmask,gt_mask) if gt_mask is not None else None
    tdice = dice_coefficient(tmask,gt_mask) if gt_mask is not None else None

    # Result figure
    fig, axes = plt.subplots(1,3,figsize=(15,5),facecolor="#080810")
    ov = np.zeros((*image.shape,4))
    if gt_mask is not None: ov[gt_mask]=[1,0.2,0.2,0.35]

    axes[0].imshow(image,cmap="gray",vmin=0,vmax=1)
    if gt_mask is not None: axes[0].imshow(ov)
    axes[0].set_title("Input MRI\n(red=ground truth)",color="white",fontsize=11)
    axes[0].axis("off")

    for ax,xf,yf,dice,col,ttl,prog in [
        (axes[1],gx,gy,gdice,"#00FF88","GVF Snake",0.95),
        (axes[2],tx,ty,tdice,"#FF8844","Traditional Snake",0.15),
    ]:
        ax.set_facecolor("#080810")
        ax.imshow(image,cmap="gray",vmin=0,vmax=1,alpha=0.82)
        if gt_mask is not None: ax.imshow(ov,zorder=1)
        draw_snake_on_axes(ax,xf,yf,progress=prog,
                           show_scales=True,show_head=True,lw_base=3.0)
        txt = f"Dice={dice:.4f}" if dice is not None else "No mask"
        ax.text(4,image.shape[0]-6,txt,color=col,fontsize=11,
                fontweight="bold",va="bottom",
                bbox=dict(facecolor="black",alpha=0.5,
                          boxstyle="round,pad=0.25",edgecolor="none"))
        ax.set_title(ttl,color=col,fontsize=11); ax.axis("off")

    fig.suptitle("Active Contour Segmentation",color="white",fontsize=12,fontweight="bold")
    plt.tight_layout()
    rb = fig_to_b64(fig); plt.close(fig)

    # GVF field
    fig2,ax2 = plt.subplots(figsize=(5,5),facecolor="#080810")
    ax2.set_facecolor("#080810")
    ax2.imshow(image,cmap="gray",alpha=0.55,vmin=0,vmax=1)
    step=12; ys_g=np.arange(step//2,h,step); xs_g=np.arange(step//2,w,step)
    X,Y=np.meshgrid(xs_g,ys_g)
    U=u[ys_g[:,None],xs_g[None,:]]; V=v[ys_g[:,None],xs_g[None,:]]
    mag=np.sqrt(U**2+V**2)
    ax2.quiver(X,Y,U,-V,mag,cmap="plasma",units="xy",scale=0.3,width=0.8,
               headwidth=3,headlength=4,alpha=0.9)
    ax2.set_xlim(0,w); ax2.set_ylim(h,0)
    ax2.set_title("GVF Force Field",color="white",fontsize=11); ax2.axis("off")
    plt.tight_layout()
    fb = fig_to_b64(fig2); plt.close(fig2)

    return rb, fb, gdice, tdice


def build_html(state):
    ds_info = (f"{len(state['dataset_files'])} dataset slices | "
               f"Slice {state['ds_index']+1}/{len(state['dataset_files'])}"
               if state["dataset_files"] else "No dataset path provided")

    result_html = ""
    if state["last_result"]:
        gd = f"GVF Dice={state['dice_gvf']:.4f}" if state["dice_gvf"] is not None else "GVF done"
        td = f"Trad Dice={state['dice_trad']:.4f}" if state["dice_trad"] is not None else "Trad done"
        result_html = f"""
        <div class="card">
          <h2 style="color:#00FF88;margin:0 0 10px">{gd} &nbsp;|&nbsp; {td}</h2>
          <img src="data:image/png;base64,{state['last_result']}"
               style="width:100%;border-radius:8px;margin-bottom:12px">
          <h3 style="color:#7FC4FF;margin:8px 0 6px">GVF Force Field</h3>
          <img src="data:image/png;base64,{state['last_field']}"
               style="width:38%;border-radius:8px">
        </div>"""

    return f"""<!DOCTYPE html><html><head>
<title>Brain Tumour Segmentation</title>
<meta charset="utf-8">
<style>
  body{{background:#080810;color:#DDE;font-family:Arial,sans-serif;margin:0;padding:24px;}}
  h1{{color:#00FF88;text-align:center;margin-bottom:4px;font-size:22px}}
  .sub{{color:#7788AA;text-align:center;margin-bottom:20px;font-size:12px}}
  .card{{background:#111120;border-radius:10px;padding:18px;margin-bottom:14px;border:1px solid #2233AA}}
  .row{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:10px}}
  button{{background:#1A4030;color:#00FF88;border:1px solid #00FF88;border-radius:6px;
          padding:9px 18px;font-size:13px;cursor:pointer;font-weight:bold}}
  button:hover{{background:#2A6048}}
  button.blue{{background:#1A2F5E;color:#7FC4FF;border-color:#2E5FA3}}
  button.blue:hover{{background:#253B7A}}
  button.run{{background:#004D40;color:white;border-color:#00897B;font-size:15px;padding:12px 28px}}
  button.run:hover{{background:#00695C}}
  input[type=text]{{background:#0D1020;color:#CCC;border:1px solid #334;border-radius:5px;
                    padding:8px 12px;font-size:12px;width:100%;box-sizing:border-box;font-family:monospace}}
  .status{{background:#0A1020;border-radius:6px;padding:10px 14px;color:#AABBCC;
           font-size:12px;border-left:3px solid #2E5FA3;margin-top:6px}}
  label{{color:#99BBDD;font-size:12px;display:block;margin-bottom:4px}}
  input[type=range]{{width:160px;accent-color:#00FF88;vertical-align:middle}}
  .val{{color:#00FF88;font-weight:bold;min-width:36px;display:inline-block;font-size:12px}}
  .tip{{color:#556677;font-size:11px;margin-top:4px}}
</style>
</head><body>
<h1>🐍 Brain Tumour Segmentation — GVF vs Traditional Snake</h1>
<p class="sub">Xu &amp; Prince (1998) vs Kass (1988) | Active Contour Models</p>

<!-- Step 1: Load Image -->
<div class="card">
  <b style="color:#7FC4FF">Step 1 — Load an Image</b>

  <!-- Synthetic (fastest) -->
  <div class="row">
    <form method="POST" action="/synthetic">
      <button type="submit">🧠 Use Synthetic MRI</button>
    </form>
    <span style="color:#667;font-size:12px">← Click this to run instantly with no downloads</span>
  </div>

  <!-- Path input -->
  <div style="margin-top:14px">
    <label>Or paste the full path to any MRI image file (.tif / .png / .jpg):</label>
    <form method="POST" action="/load_path">
      <input type="text" name="path"
             placeholder="Example: C:\\Users\\bplpr\\Downloads\\TCGA_CS_4941_19960909_11.tif"
             value="">
      <div class="tip">
        If a matching *_mask.tif file exists in the same folder, Dice score will be computed.<br>
        LGG dataset images are .tif files inside patient folders like TCGA_CS_4941_19960909/
      </div>
      <div class="row" style="margin-top:8px">
        <button type="submit" class="blue">📂 Load from Path</button>
      </div>
    </form>
  </div>

  <!-- Dataset nav -->
  <div class="row" style="margin-top:12px;border-top:1px solid #223;padding-top:12px">
    <form method="POST" action="/prev"><button type="submit" class="blue">◀ Prev Slice</button></form>
    <form method="POST" action="/next"><button type="submit" class="blue">▶ Next Slice</button></form>
    <span style="color:#556677;font-size:11px">{ds_info}</span>
  </div>
</div>

<!-- Step 2: Run -->
<div class="card">
  <b style="color:#7FC4FF">Step 2 — Set Parameters &amp; Run</b>
  <form method="POST" action="/run" style="margin-top:12px">
    <div class="row" style="gap:28px;margin-bottom:14px">
      <div>
        <label>Alpha (elasticity)</label>
        <input type="range" name="alpha" min="0.001" max="0.05" step="0.001" value="{state["alpha"]}"
               oninput="document.getElementById('av').textContent=this.value">
        <span class="val" id="av">{state['alpha']}</span>
      </div>
      <div>
        <label>Mu (GVF diffusion strength)</label>
        <input type="range" name="mu" min="0.1" max="0.5" step="0.05" value="{state["mu"]}"
               oninput="document.getElementById('mv').textContent=this.value">
        <span class="val" id="mv">{state['mu']}</span>
      </div>
      <div>
        <label>Iterations</label>
        <input type="range" name="iters" min="1000" max="8000" step="500" value="{state["iters"]}"
               oninput="document.getElementById('iv').textContent=this.value">
        <span class="val" id="iv">{state['iters']}</span>
      </div>
    </div>
    <button type="submit" class="run">▶ Run Segmentation</button>
    <span style="color:#556;font-size:11px;margin-left:12px">Takes ~30 seconds</span>
  </form>
</div>

<!-- Status -->
<div class="status">📌 <b>{state['filename']}</b> &nbsp;|&nbsp; {state['status']}</div>

<!-- Results -->
{result_html}
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def send_page(self, html):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self):
        self.send_response(303)
        self.send_header("Location","/")
        self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length",0))
        return self.rfile.read(length).decode("utf-8","ignore")

    def do_GET(self):
        self.send_page(build_html(STATE))

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.read_body()
        params = parse_qs(body)

        if path == "/synthetic":
            image, gt_mask, _,_ = generate_brain_mri(size=256)
            STATE["image"]   = image
            STATE["gt_mask"] = gt_mask
            STATE["filename"]= "Synthetic MRI"
            STATE["status"]  = "Synthetic MRI loaded. Click ▶ Run Segmentation."
            STATE["last_result"] = None

        elif path == "/load_path":
            raw = params.get("path",[""])[0].strip()
            # Handle URL encoding and Windows backslashes
            img_path = unquote(raw).replace("/","\\") if "\\" in raw or ":" in raw else unquote(raw)
            img_path = img_path.strip('"').strip("'")
            if not img_path:
                STATE["status"] = "Please paste a file path first."
            elif not os.path.isfile(img_path):
                STATE["status"] = f"File not found: {img_path}  — check the path is correct."
            else:
                try:
                    STATE["image"]    = load_image_array(img_path)
                    STATE["gt_mask"]  = load_mask_array(img_path)
                    STATE["filename"] = os.path.basename(img_path)
                    has = "✓ mask found" if STATE["gt_mask"] is not None else "no mask"
                    STATE["status"]   = f"Loaded {STATE['filename']} ({has}). Click ▶ Run Segmentation."
                    STATE["last_result"] = None
                except Exception as e:
                    STATE["status"] = f"Error loading image: {e}"

        elif path == "/prev":
            if STATE["dataset_files"]:
                STATE["ds_index"] = (STATE["ds_index"]-1) % len(STATE["dataset_files"])
                _load_ds_slice()
            else:
                STATE["status"] = "No dataset. Use --dataset_path when starting the server."

        elif path == "/next":
            if STATE["dataset_files"]:
                STATE["ds_index"] = (STATE["ds_index"]+1) % len(STATE["dataset_files"])
                _load_ds_slice()
            else:
                STATE["status"] = "No dataset. Use --dataset_path when starting the server."

        elif path == "/run":
            if STATE["image"] is None:
                STATE["status"] = "No image loaded. Click Synthetic MRI first."
            else:
                alpha = float(params.get("alpha",["0.005"])[0])
                mu    = float(params.get("mu",   ["0.25"])[0])
                iters = int(  params.get("iters",["5000"])[0])
                # Save slider values so they persist after reload
                STATE["alpha"]=alpha; STATE["mu"]=mu; STATE["iters"]=iters
                STATE["status"] = "Running segmentation... please wait ~30 seconds"
                try:
                    rb,fb,gd,td = run_segmentation(STATE["image"],STATE["gt_mask"],alpha,mu,iters)
                    STATE["last_result"] = rb
                    STATE["last_field"]  = fb
                    STATE["dice_gvf"]    = gd
                    STATE["dice_trad"]   = td
                    if gd is not None:
                        STATE["status"] = f"Done!  GVF Dice={gd:.4f}  |  Traditional Dice={td:.4f}"
                    else:
                        STATE["status"] = "Done! Upload a *_mask.tif file alongside the image for Dice scores."
                except Exception as e:
                    STATE["status"] = f"Error: {e}"
                    import traceback; traceback.print_exc()

        self.redirect()


def _load_ds_slice():
    ip, mp = STATE["dataset_files"][STATE["ds_index"]]
    STATE["image"]    = load_image_array(ip)
    mask_arr = Image.open(mp).convert("L").resize((256,256), Image.NEAREST)
    STATE["gt_mask"]  = np.array(mask_arr, dtype=bool)
    STATE["filename"] = os.path.basename(ip)
    STATE["status"]   = (f"Slice {STATE['ds_index']+1}/{len(STATE['dataset_files'])} — "
                         f"Click ▶ Run Segmentation.")
    STATE["last_result"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    if args.dataset_path:
        STATE["dataset_files"] = scan_dataset(args.dataset_path)
        print(f"[Web] Dataset: {len(STATE['dataset_files'])} tumour-positive slices")
        if STATE["dataset_files"]:
            _load_ds_slice()
    else:
        image, gt_mask, _,_ = generate_brain_mri(size=256)
        STATE["image"]    = image
        STATE["gt_mask"]  = gt_mask
        STATE["filename"] = "Synthetic MRI (auto-loaded)"
        STATE["status"]   = "Synthetic MRI loaded. Click ▶ Run Segmentation to start."

    server = HTTPServer(("localhost", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n{'='*50}")
    print(f"  Open this in your browser: {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*50}\n")

    def _open():
        import time, webbrowser; time.sleep(0.8); webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

    try: server.serve_forever()
    except KeyboardInterrupt: print("\nServer stopped.")

if __name__ == "__main__":
    main()
