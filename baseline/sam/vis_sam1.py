#!/usr/bin/env python3
"""
visualize_sam_masks.py

Reads label masks from:   pwd/sam_out/{k}_mask.png   (uint16 ids from run_sam.py)
Original images from:     pwd/parent/input/{k}.png
Writes color renders to:  pwd/sam_out/{k}_color.png   (each id a distinct color)
                          pwd/sam_out/{k}_blend.png   (color overlaid on original)

Background id = 0 stays transparent.
"""

from pathlib import Path
import cv2
import numpy as np

PWD = Path.cwd()
INPUT_DIR = PWD.parent / "input"
SAM_OUT   = PWD / "sam_out"
SAM_OUT.mkdir(parents=True, exist_ok=True)

def id_to_color(idx: int) -> tuple:
    """
    Deterministic bright color for a given id (OpenCV BGR).
    Uses HSV wheel; skips idx=0 (background).
    """
    if idx <= 0:
        return (0, 0, 0)
    # Spread hues deterministically
    hue = (37 * idx) % 180      # OpenCV hue range [0,180)
    sat = 200
    val = 255
    hsv = np.uint8([[[hue, sat, val]]])  # 1x1 HSV
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def colorize_label(label: np.ndarray) -> np.ndarray:
    """
    Convert uint16 label map to BGR color image (background black).
    """
    assert label.ndim == 2, "label must be HxW single-channel"
    h, w = label.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    ids = np.unique(label)
    ids = ids[ids != 0]  # skip background
    for idx in ids:
        mask = (label == idx)
        color = id_to_color(int(idx))
        color_img[mask] = color
    return color_img

def alpha_blend(base_bgr: np.ndarray, overlay_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Alpha-blend overlay onto base where overlay != 0 (non-background).
    """
    mask = np.any(overlay_bgr != 0, axis=2)  # HxW bool
    blended = base_bgr.copy()
    # Only blend on mask==True to keep background untouched
    blended[mask] = (alpha * overlay_bgr[mask] + (1 - alpha) * base_bgr[mask]).astype(np.uint8)
    return blended

def main():
    label_paths = sorted(SAM_OUT.glob("*_mask.png"))
    if not label_paths:
        raise FileNotFoundError(f"No *_mask.png found in {SAM_OUT}")

    for lp in label_paths:
        stem = lp.name.replace("_mask.png", "")
        # Read label as-is (uint16)
        label = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED)
        if label is None or label.ndim != 2:
            print(f"[WARN] Skipping {lp} (not a valid single-channel mask).")
            continue

        color = colorize_label(label)
        color_path = SAM_OUT / f"{stem}_color.png"
        cv2.imwrite(str(color_path), color)

        # Try to load original image for blending
        src_img_path = INPUT_DIR / f"{stem}.png"
        if src_img_path.exists():
            orig = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
            if orig is not None and orig.shape[:2] == color.shape[:2]:
                blend = alpha_blend(orig, color, alpha=0.45)
                blend_path = SAM_OUT / f"{stem}_blend.png"
                cv2.imwrite(str(blend_path), blend)
                print(f"[OK] {stem}: wrote {color_path.name}, {blend_path.name}")
                continue

        print(f"[OK] {stem}: wrote {color_path.name} (original not found or size mismatch)")

if __name__ == "__main__":
    main()
