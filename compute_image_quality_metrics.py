# Metrics from processed images made by your pipeline
# Requires: pip install opencv-python pandas numpy scikit-image

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import shannon_entropy

# --- paths ---
OUT_ROOT = r"C:\Users\engsa\Desktop\said_AI_proc"          # folder that contains manifest.csv
MANIFEST = os.path.join(OUT_ROOT, "manifest.csv")
METRICS_CSV = os.path.join(OUT_ROOT, "metrics.csv")

# --- settings ---
GLARE_THRESH = 250          # on 8-bit _norm images
MIN_BG_DISTANCE = 25        # px away from defect mask for background (uses distance transform)
BG_MAX_FRAC = 0.08          # cap background area fraction (use smoothest/lowest-response parts)
LAPLACIAN_KSIZE = 3

def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img = (img - img.min()) / max(1e-6, (img.max() - img.min()))
    return (255*img).astype(np.uint8)

def variance_of_laplacian(gray_u8):
    lap = cv2.Laplacian(gray_u8, cv2.CV_32F, ksize=LAPLACIAN_KSIZE)
    return float(np.var(lap))

def compute_metrics(norm_path, mask_path):
    # read
    norm = cv2.imread(norm_path, cv2.IMREAD_UNCHANGED)
    if norm is None:
        raise RuntimeError(f"Can't read {norm_path}")
    if norm.ndim == 3:
        norm = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    norm = to_uint8(norm)

    # mask (optional but recommended)
    mask = None
    if mask_path and Path(mask_path).exists():
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            mask = (m > 0)

    h, w = norm.shape

    # if no mask, make a very simple center ROI so code never breaks
    if mask is None or mask.sum() == 0:
        mask = np.zeros_like(norm, dtype=bool)
        r1, r2 = h//3, 2*h//3
        c1, c2 = w//3, 2*w//3
        mask[r1:r2, c1:c2] = True

    # glare mask on _norm
    glare = norm > GLARE_THRESH

    # background: pixels far from defect and not glare
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    bg = (dist >= MIN_BG_DISTANCE) & (~glare)

    # if background is too large, keep only a small/smooth subset (lowest intensities)
    bg_idx = np.flatnonzero(bg)
    target = int(BG_MAX_FRAC * norm.size)
    if bg_idx.size > 3*max(1, target):
        # use low-intensity pixels as smoother background
        vals = norm.flat[bg_idx]
        order = np.argsort(vals)  # ascending
        keep = bg_idx[order[:target]]
        bg = np.zeros_like(bg, dtype=bool)
        bg.flat[keep] = True

    # compute stats
    defect_vals = norm[mask]
    bg_vals = norm[bg] if bg.any() else norm[~mask]

    mu_def = float(np.mean(defect_vals)) if defect_vals.size else np.nan
    mu_bg  = float(np.mean(bg_vals))     if bg_vals.size     else np.nan
    sd_bg  = float(np.std(bg_vals))      if bg_vals.size     else np.nan

    cnr = float(abs(mu_def - mu_bg) / sd_bg) if sd_bg and sd_bg > 0 else np.nan
    snr = float(np.mean(norm) / (np.std(norm) + 1e-9))
    sharp = variance_of_laplacian(norm)
    entr = float(shannon_entropy(norm))
    glare_ratio = float(np.mean(glare))

    # defect area %
    defect_area_pct = 100.0 * (np.sum(mask) / norm.size)

    return {
        "mean_defect": mu_def,
        "mean_bg": mu_bg,
        "std_bg": sd_bg,
        "CNR": cnr,
        "SNR": snr,
        "sharpness_varLapl": sharp,
        "entropy": entr,
        "glare_ratio": glare_ratio,
        "defect_area_pct": defect_area_pct
    }

def main():
    df = pd.read_csv(MANIFEST)
    rows = []

    for i, r in df.iterrows():
        norm_path = r.get("norm", "")
        mask_path = r.get("mask", "")

        if not isinstance(norm_path, str) or not Path(norm_path).exists():
            print(f"Skip (no norm): {r.get('original','')}")
            continue

        try:
            m = compute_metrics(norm_path, mask_path if isinstance(mask_path, str) else "")
        except Exception as e:
            print(f"Error on {norm_path}: {e}")
            continue

        # carry useful columns from manifest if present
        row = {
            "file": r.get("original", ""),
            "material": r.get("material", ""),
            "condition": r.get("condition", ""),
            "angle_deg": r.get("angle_deg", ""),
            "wavelength": r.get("wavelength", ""),
            "norm_path": norm_path,
            "mask_path": mask_path
        }
        row.update(m)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(METRICS_CSV, index=False)
    print(f"\nSaved metrics to: {METRICS_CSV}")
    if not out.empty:
        # quick summaries
        print("\nTop 10 images by CNR:")
        print(out.sort_values("CNR", ascending=False).head(10)[["file","CNR","material","angle_deg","wavelength"]])

        print("\nAverage CNR by angle & wavelength:")
        print(out.groupby(["angle_deg","wavelength"])["CNR"].mean().round(2))

if __name__ == "__main__":
    main()
