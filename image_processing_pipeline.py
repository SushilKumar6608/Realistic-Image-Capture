"""
Batch BMP image processing:
Generates _norm, _mask, _fused, and _overlay versions
for all .bmp images (recursively) and saves them into
an organized folder tree with a manifest.csv file.

Structure:
    <out_root>/<material>/<condition>/<angle>/<wavelength>/
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.morphology import remove_small_holes
from skimage.measure import label, regionprops

# ---------- USER CONFIG ----------
IN_ROOT  = r"C:\Users\engsa\Desktop\said AI"        # Input root folder
OUT_ROOT = r"C:\Users\engsa\Desktop\said_AI_proc"   # Output root folder
# --------------------------------

# Defect mask filter parameters
MIN_AREA, MAX_AREA = 300, 200_000
MIN_ASPECT, MIN_ECC = 2.0, 0.90
GLARE_THRESH8 = 250  # glare threshold (uint8)

# Regex for structured filenames
META_RE = re.compile(
    r'^(metal|plastic|transparent)_(defected|undefected)_(\d+)deg_(red|green|blue|white)$',
    re.IGNORECASE
)

def parse_meta(stem: str):
    """Extract material, condition, angle, and wavelength from filename."""
    m = META_RE.match(stem)
    if not m:
        return None
    material, condition, angle, wavelength = m.groups()
    return {
        "material": material.lower(),
        "condition": condition.lower(),
        "angle_deg": int(angle),
        "wavelength": wavelength.lower()
    }

def ensure_uint8(gray):
    """Convert image to 8-bit grayscale (0–255)."""
    if gray.dtype == np.uint8:
        return gray
    g = gray.astype(np.float32)
    g = (g - g.min()) / max(1e-6, (g.max() - g.min()))
    return (255 * g).astype(np.uint8)

def make_norm(gray_u8):
    """Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray_u8)

def defect_response(gray_u8):
    """Enhancement map to highlight scratches."""
    blur = cv2.GaussianBlur(gray_u8, (0,0), 25)
    flat = cv2.normalize(cv2.subtract(gray_u8, blur), None, 0, 255, cv2.NORM_MINMAX)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    toph = cv2.morphologyEx(flat, cv2.MORPH_TOPHAT, se)
    both = cv2.morphologyEx(flat, cv2.MORPH_BLACKHAT, se)
    gx = cv2.Sobel(flat, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(flat, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    resp = cv2.addWeighted(toph, 0.45, both, 0.45, 0)
    resp = cv2.addWeighted(resp, 1.0, grad, 0.35, 0)
    return cv2.GaussianBlur(resp, (3,3), 1)

def make_mask(resp_u8, gray_u8):
    """Binary defect mask from response map."""
    _, bw = cv2.threshold(resp_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))
    glare = (gray_u8 > GLARE_THRESH8).astype(np.uint8) * 255
    bw[glare == 255] = 0

    lab = label(bw > 0, connectivity=2)
    out = np.zeros_like(bw, dtype=bool)
    for r in regionprops(lab):
        a = r.area
        if a < MIN_AREA or a > MAX_AREA:
            continue
        maj = max(1.0, r.major_axis_length)
        minax = max(1.0, r.minor_axis_length)
        if maj / minax < MIN_ASPECT:
            continue
        if r.eccentricity < MIN_ECC:
            continue
        out[lab == r.label] = True

    out = remove_small_holes(out, area_threshold=200)
    return (out.astype(np.uint8) * 255)

def make_fused(norm_u8, resp_u8):
    """Combine normalized image and response map."""
    fused = cv2.addWeighted(norm_u8, 0.6, resp_u8, 0.6, 0)
    return cv2.normalize(fused, None, 0, 255, cv2.NORM_MINMAX)

def make_overlay(bgr, mask_u8, color=(40,255,40), alpha=0.45):
    """Overlay mask on image."""
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    mask = (mask_u8 > 0)[:, :, None]
    color_arr = np.zeros_like(bgr)
    color_arr[:] = color
    overlay = np.where(mask, cv2.addWeighted(bgr, 1-alpha, color_arr, alpha, 0), bgr)
    cnts, _ = cv2.findContours((mask_u8>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0,255,255), 1)
    return overlay

def process_one(in_path: Path, out_root: Path, manifest_rows: list):
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️ Skipping unreadable: {in_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray8 = ensure_uint8(gray)
    stem = in_path.stem
    meta = parse_meta(stem)

    if meta:
        out_dir = out_root / meta["material"] / meta["condition"] / f'{meta["angle_deg"]}deg' / meta["wavelength"]
    else:
        out_dir = out_root / "unparsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = make_norm(gray8)
    resp = defect_response(gray8)
    mask = make_mask(resp, gray8)
    fused = make_fused(norm, resp)
    over = make_overlay(img if img.ndim == 3 else cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR), mask)

    paths = {
        "original": out_dir / f"{stem}.bmp",
        "norm": out_dir / f"{stem}_norm.png",
        "mask": out_dir / f"{stem}_mask.png",
        "fused": out_dir / f"{stem}_fused.png",
        "overlay": out_dir / f"{stem}_overlay.png",
    }

    cv2.imwrite(str(paths["original"]), img)
    cv2.imwrite(str(paths["norm"]), norm)
    cv2.imwrite(str(paths["mask"]), mask)
    cv2.imwrite(str(paths["fused"]), fused)
    cv2.imwrite(str(paths["overlay"]), over)

    manifest_rows.append({
        "in_path": str(in_path),
        "out_dir": str(out_dir),
        "original": str(paths["original"]),
        "norm": str(paths["norm"]),
        "mask": str(paths["mask"]),
        "fused": str(paths["fused"]),
        "overlay": str(paths["overlay"]),
        **(meta if meta else {"material": "", "condition": "", "angle_deg": "", "wavelength": ""})
    })

def main():
    in_root = Path(IN_ROOT)
    out_root = Path(OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # 🟩 Only process .bmp files (case-insensitive)
    bmp_files = [p for p in in_root.rglob("*.bmp")]
    bmp_files.sort()
    print(f"Found {len(bmp_files)} BMP images under {in_root}\n")

    rows = []
    for i, p in enumerate(bmp_files, 1):
        print(f"[{i:>3}/{len(bmp_files)}] Processing {p.name}")
        process_one(p, out_root, rows)

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "manifest.csv", index=False)
    print(f"\n✅ Saved manifest: {out_root / 'manifest.csv'}")
    print("🎯 Done.")

if __name__ == "__main__":
    main()
