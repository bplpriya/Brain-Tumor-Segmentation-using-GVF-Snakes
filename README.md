# Brain Tumour Segmentation — Active Contour Models (v2)
### CV Project — GVF vs Traditional Snake Comparison
**Student:** Bala Pavani Lakshmi Priya Koppuravuri | Panther ID: 002895824

---

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Narrative | Traditional (close) vs GVF (far) — wrong | **Both start SAME far distance** — correct |
| Contour art | Plain circle line | **Real snake** with head, eyes, tongue, scales |
| GUI | None | **Full interactive GUI** — upload images, live results |
| Dataset | Synthetic only | **Real LGG Kaggle MRI dataset** supported |
| Evaluation | Single image | **Batch evaluation** with train/test split by patient |

---

## Quick Start

```bash
cd brain_tumor_v2/
pip install numpy scipy matplotlib scikit-image pillow

# 1. Run artistic comparison (GVF vs Traditional, same far init)
python comparison_fixed.py

# 2. Launch interactive GUI (requires desktop / display)
python gui_app.py

# 3. GUI with real dataset
python gui_app.py --dataset_path /path/to/lgg-mri-segmentation

# 4. Batch evaluate on real LGG dataset
python evaluate_dataset.py --dataset_path /path/to/dataset --n_eval 30
```

---

## Getting the Real Dataset

```python
import kagglehub
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
print("Path to dataset files:", path)
# Then run:
# python gui_app.py --dataset_path <path>
# python evaluate_dataset.py --dataset_path <path>
```

---

## File Structure

```
brain_tumor_v2/
├── synthetic_mri.py      # Synthetic brain MRI generator
├── snake_traditional.py  # Traditional snake (matrix + scikit-image)
├── gvf.py                # GVF diffusion field + GVF snake
├── metrics.py            # Dice, Hausdorff, Precision/Recall
├── snake_art.py          # 🆕 Artistic snake rendering (head/scales/tongue)
├── dataset.py            # 🆕 LGG Kaggle dataset loader + train/test split
├── comparison_fixed.py   # 🆕 Correct GVF vs Trad (SAME far init)
├── evaluate_dataset.py   # 🆕 Batch evaluation on real dataset
├── gui_app.py            # 🆕 Full interactive GUI
└── visualization.py      # GVF field, energy plots, etc.
```

---

## The Correct Narrative (Why GVF is Better)

**Wrong approach (v1):**
- Traditional: starts 8px away from tumour → Dice = 0.98 ✓
- GVF: starts 22px away from tumour → Dice = 0.53 ✗
- Conclusion: "Traditional is better" ← WRONG

**Correct approach (v2):**
- Traditional: starts **38px away** → gets stuck at oedema boundary (Dice ~0.3)
- GVF: starts **same 38px away** → diffused forces reach tumour (Dice ~0.7+)
- Conclusion: "GVF has LARGER CAPTURE RANGE" ← Correct

This matches Xu & Prince (1998): the key advantage of GVF is that it propagates
edge information across the image via diffusion, allowing convergence from
initialisation points that traditional snakes cannot handle.

---

## Mathematical Summary

### Traditional Snake Energy (Kass 1988)
```
E_snake = ∫ [α|v'(s)|² + β|v''(s)|²] ds  +  ∫ E_ext(v(s)) ds
(A + γI) · x^{t+1} = γ · x^t + f_x(x^t, y^t)
```

### GVF Diffusion (Xu & Prince 1998)
```
u^{n+1} = (1 − b·Δt) · u^n  +  μ·Δt · ∇²u^n  +  b·Δt · f_x
v^{n+1} = (1 − b·Δt) · v^n  +  μ·Δt · ∇²v^n  +  b·Δt · f_y

where b = |∇f|²  (edge strength weighting)
      μ = regularisation (diffusion strength)
```
**Key:** Large μ → forces diffuse far from edges → large capture range

### Dice Coefficient
```
DSC = 2|A ∩ B| / (|A| + |B|)    ∈ [0,1]
```

---

## GUI Guide

```
┌─────────────────────────────────────────────┬──────────────────┐
│  Input MRI  |  GVF Result  |  Trad Result   │  🐍 GVF Brain    │
│  (GT=red)   |  (snake art) |  (snake art)   │    Tumour GUI    │
├─────────────┬──────────────┬────────────────│  [📁 Load Image] │
│  GVF Force  │   Metrics    │   Status       │  [🧠 Synthetic]  │
│   Field     │   Bar Chart  │   Messages     │  [◀ Prev Slice]  │
└─────────────┴──────────────┴────────────────│  [▶ Next Slice]  │
                                              │  [▶ Run Seg.]    │
                                              │  [🎬 Save Anim]  │
                                              │  [💾 Save Res.]  │
                                              │  ─── Sliders ─── │
                                              │  α (elasticity)  │
                                              │  μ (GVF diff.)   │
                                              │  Iterations      │
                                              └──────────────────┘
```

---

## References

1. Kass, M., Witkin, A., & Terzopoulos, D. (1988). Snakes: Active contour models. *IJCV*, 1(4), 321–331.
2. Xu, C., & Prince, J. L. (1998). Snakes, shapes, and gradient vector flow. *IEEE TIP*, 7(3), 359–369.
3. Buda, M., Saha, A., & Mazurowski, M. A. (2019). Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm. *Computers in Biology and Medicine*.
