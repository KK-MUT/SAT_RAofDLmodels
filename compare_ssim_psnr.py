#!/usr/bin/env python3
"""
compare_ssim_psnr.py — Compute SSIM / PSNR / MSE between reference and test images.

Typical use-cases
-----------------
1) Compare an original dataset vs. its restored / adversarially perturbed version
2) Evaluate per-image SSIM/PSNR and export a CSV summary
3) Quickly compare two single images

Examples
--------
# Single pair
python compare_ssim_psnr.py --ref img_orig.png --test img_adv.png

# Two folders (same filenames, recursive, supports .jpg/.png/.tif)
python compare_ssim_psnr.py --ref ./clean/ --test ./adv/ --ext .png --out metrics.csv

# Save visual diff-maps as well
python compare_ssim_psnr.py --ref ./A/ --test ./B/ --save-diff --out diff_metrics.csv
"""

import argparse
from pathlib import Path
import os
import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse


# ----------------------------- Helpers -----------------------------
def load_image_gray(path: str) -> np.ndarray:
    """Load image as grayscale float32 in [0,1]."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compare_pair(ref_path: str, test_path: str):
    """Compute SSIM, PSNR, MSE for a single pair of images."""
    ref = load_image_gray(ref_path)
    test = load_image_gray(test_path)

    # Resize test to match ref if necessary
    if ref.shape != test.shape:
        from skimage.transform import resize
        test = resize(test, ref.shape, order=1, anti_aliasing=True)

    ssim_val = ssim(ref, test, data_range=1.0)
    psnr_val = psnr(ref, test, data_range=1.0)
    mse_val  = mse(ref, test)
    return ssim_val, psnr_val, mse_val


def save_diff_image(ref_path: str, test_path: str, out_path: Path):
    """Save an absolute-difference heatmap between reference and test images."""
    ref = load_image_gray(ref_path)
    test = load_image_gray(test_path)

    # Resize if shapes differ
    if ref.shape != test.shape:
        from skimage.transform import resize
        test = resize(test, ref.shape, order=1, anti_aliasing=True)

    diff = np.abs(ref - test)
    diff = (diff / diff.max() * 255.0).astype(np.uint8)
    Image.fromarray(diff).save(out_path)


def list_images(folder: str, ext_filter: str):
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    return sorted([str(p) for p in Path(folder).rglob("*"+ext_filter) if p.suffix.lower() in exts])


# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Compute SSIM / PSNR / MSE between reference and test images")
    ap.add_argument("--ref", required=True, help="Reference image or folder")
    ap.add_argument("--test", required=True, help="Test image or folder to compare with reference")
    ap.add_argument("--ext", type=str, default="", help="File extension filter (e.g., .png) when comparing folders")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path")
    ap.add_argument("--save-diff", action="store_true", help="Save per-pair difference heatmaps next to CSV")
    args = ap.parse_args()

    ref_path = Path(args.ref)
    test_path = Path(args.test)

    # --- Single pair ---
    if ref_path.is_file() and test_path.is_file():
        s, p, m = compare_pair(str(ref_path), str(test_path))
        print(f"File pair:\n  {ref_path}\n  {test_path}")
        print(f"SSIM={s:.4f}  PSNR={p:.2f} dB  MSE={m:.6f}")
        return

    # --- Folder vs Folder ---
    if ref_path.is_dir() and test_path.is_dir():
        ref_imgs = list_images(str(ref_path), args.ext)
        test_imgs = list_images(str(test_path), args.ext)

        # Build mapping by filename (no directories)
        test_map = {Path(p).name: p for p in test_imgs}
        if not ref_imgs:
            print("No images found in reference folder.")
            return

        rows = []
        out_dir = Path(args.out).parent if args.out else Path("./")
        for rp in ref_imgs:
            fname = Path(rp).name
            if fname not in test_map:
                print(f"[WARN] Missing counterpart for {fname}")
                continue
            tp = test_map[fname]
            s, p, m = compare_pair(rp, tp)
            rows.append((fname, s, p, m))
            print(f"{fname:30s}  SSIM={s:.4f}  PSNR={p:.2f} dB  MSE={m:.6f}")

            if args.save_diff:
                diff_path = out_dir / f"diff_{fname}"
                save_diff_image(rp, tp, diff_path)

        if args.out:
            import csv
            with open(args.out, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "SSIM", "PSNR_dB", "MSE"])
                for r in rows:
                    writer.writerow(r)
            print(f"\nMetrics saved to: {args.out}")
        return

    raise ValueError("Please provide either two files or two folders for comparison.")


if __name__ == "__main__":
    main()
