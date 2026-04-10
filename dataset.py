"""
dataset.py
----------
Loader for the LGG MRI Segmentation Dataset (Kaggle)
Dataset: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Usage:
    import kagglehub
    path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

    from dataset import LGGDataset
    ds = LGGDataset(path)
    train, test = ds.split(test_ratio=0.2)

Dataset structure on disk:
    kaggle_3m/
        TCGA_<patient_id>/
            TCGA_<id>_<slice>.tif       ← MRI slice
            TCGA_<id>_<slice>_mask.tif  ← binary tumour mask
"""

import os, glob, random
import numpy as np
from PIL import Image


class LGGDataset:
    """
    Loads MRI slices + corresponding tumour masks from the LGG dataset.

    Parameters
    ----------
    root_dir   : path returned by kagglehub.dataset_download(...)
    size       : resize all images to (size, size) — default 256
    only_tumor : if True, only include slices that contain tumour pixels
    seed       : random seed for reproducible splitting
    """

    def __init__(self, root_dir: str, size: int = 256,
                 only_tumor: bool = True, seed: int = 42):
        self.root_dir   = root_dir
        self.size       = size
        self.only_tumor = only_tumor
        self.seed       = seed
        self.samples    = self._scan()
        print(f"[LGGDataset] Found {len(self.samples)} "
              f"{'tumour-positive ' if only_tumor else ''}slices "
              f"in {root_dir}")

    # ── internal scan ─────────────────────────────────────────────────────────

    def _scan(self):
        """Return list of (img_path, mask_path) pairs."""
        # masks are the *_mask.tif files; images are the same name without _mask
        mask_paths = sorted(glob.glob(
            os.path.join(self.root_dir, "**", "*_mask.tif"),
            recursive=True,
        ))
        if not mask_paths:
            # Also try .png / .jpg inside kaggle_3m sub-folder
            mask_paths = sorted(glob.glob(
                os.path.join(self.root_dir, "**", "*_mask.*"),
                recursive=True,
            ))

        samples = []
        for mp in mask_paths:
            ip = mp.replace("_mask", "")
            if os.path.isfile(ip):
                if self.only_tumor:
                    mask = np.array(Image.open(mp).convert("L"))
                    if mask.max() == 0:
                        continue   # skip empty masks
                samples.append((ip, mp))

        return samples

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Return (image, mask) both as float64 numpy arrays in [0,1]."""
        img_path, mask_path = self.samples[idx]
        image = self._load_gray(img_path)
        mask  = self._load_mask(mask_path)
        return image, mask, img_path

    def split(self, test_ratio: float = 0.20):
        """
        Split dataset into train / test by patient (not by slice) to
        prevent data leakage.

        Returns
        -------
        train_ds, test_ds : two LGGSubset objects
        """
        # Group slices by patient folder
        patient_groups: dict[str, list] = {}
        for ip, mp in self.samples:
            patient = os.path.basename(os.path.dirname(ip))
            patient_groups.setdefault(patient, []).append((ip, mp))

        patients = list(patient_groups.keys())
        rng = random.Random(self.seed)
        rng.shuffle(patients)

        n_test = max(1, int(len(patients) * test_ratio))
        test_patients  = set(patients[:n_test])
        train_patients = set(patients[n_test:])

        train_samples = [s for p in train_patients for s in patient_groups[p]]
        test_samples  = [s for p in test_patients  for s in patient_groups[p]]

        print(f"[LGGDataset] Train: {len(train_samples)} slices "
              f"({len(train_patients)} patients) | "
              f"Test: {len(test_samples)} slices "
              f"({len(test_patients)} patients)")

        return (LGGSubset(train_samples, self.size),
                LGGSubset(test_samples,  self.size))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_gray(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize(
            (self.size, self.size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float64)
        return arr / (arr.max() + 1e-10)

    def _load_mask(self, path: str) -> np.ndarray:
        mask = Image.open(path).convert("L").resize(
            (self.size, self.size), Image.NEAREST)
        return np.array(mask, dtype=bool)

    def get_tumor_center(self, mask: np.ndarray):
        """Return (cy, cx) centroid of tumour mask."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            h, w = mask.shape
            return h // 2, w // 2
        return int(ys.mean()), int(xs.mean())

    def get_tumor_radius(self, mask: np.ndarray):
        """Return approximate bounding radius of tumour."""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return 20
        cy, cx = ys.mean(), xs.mean()
        dists = np.sqrt((ys - cy)**2 + (xs - cx)**2)
        return float(dists.max())


class LGGSubset:
    """A pre-sliced subset of LGGDataset."""

    def __init__(self, samples: list, size: int = 256):
        self.samples = samples
        self.size    = size
        self._parent = LGGDataset.__new__(LGGDataset)
        self._parent.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, mp = self.samples[idx]
        image = self._parent._load_gray(ip)
        mask  = self._parent._load_mask(mp)
        return image, mask, ip

    def _load_gray(self, path):
        return self._parent._load_gray(path)

    def _load_mask(self, path):
        return self._parent._load_mask(path)
