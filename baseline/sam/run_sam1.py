#!/usr/bin/env python3
import os
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ----------------------------
# Config
# ----------------------------
# Input images: pwd/parent/input/0.png, 1.png, ...
INPUT_DIR = Path.cwd()
OUTPUT_DIR = Path.cwd() / "sam_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to your downloaded SAM checkpoint (ViT-H by default)
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", str(Path.cwd() / "sam_vit_h_4b8939.pth"))
SAM_MODEL_TYPE = "vit_h"  # one of: vit_h, vit_l, vit_b

# Reasonable defaults for automatic mask generation
MASKGEN_KW = dict(
    points_per_side=32,              # higher = denser proposals; 32 is a good start
    pred_iou_thresh=0.86,            # keep only good masks
    stability_score_thresh=0.92,     # stability filter
    box_nms_thresh=0.7,              # drop overlaps
    min_mask_region_area=64,         # remove tiny fragments
)

# ----------------------------
# Helpers
# ----------------------------
def _masks_to_label_png(masks, h, w):
    """
    Convert SAM mask list to a single-channel label image (uint16) where
    each mask has a unique positive id (1..N). Overlaps are resolved by
    sorting by (score, area) so higher quality masks win last.
    """
    if not masks:
        return np.zeros((h, w), dtype=np.uint16)

    # Sort so the best masks paint last (win in overlaps)
    masks_sorted = sorted(
        masks,
        key=lambda m: (float(m.get("predicted_iou", 0.0)), int(m["area"])),
        reverse=True,
    )
    label = np.zeros((h, w), dtype=np.uint16)
    for idx, m in enumerate(masks_sorted, start=1):
        label[m["segmentation"]] = idx
    return label

def _draw_contours_overlay(img_bgr, masks):
    """
    Draw mask contours on top of the image for quick visual inspection.
    """
    vis = img_bgr.copy()
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 1)
    return vis

def _rle_encode_boolmask(mask_bool):
    """
    Minimal COCO RLE for a binary mask. Uses bytes RLE via pycocotools
    if available; otherwise, a simple uncompressed RLE fallback.
    """
    try:
        from pycocotools import mask as mutils
        rle = mutils.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
        # pycocotools returns counts as bytes; to make json serializable, decode to str
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        # Fallback: simple counts RLE (not COCO-compliant compression, but JSON-serializable)
        # Flatten in column-major (as COCO expects Fortran order)
        arr = np.asfortranarray(mask_bool)
        flat = arr.reshape((-1,), order="F").astype(np.uint8)
        counts = []
        prev = 0
        run_len = 0
        for v in flat:
            if v != prev:
                counts.append(run_len)
                run_len = 1
                prev = v
            else:
                run_len += 1
        counts.append(run_len)
        return {"size": list(mask_bool.shape), "counts": counts}

# ----------------------------
# Main
# ----------------------------
def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    # Load model
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to("cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    mask_generator = SamAutomaticMaskGenerator(sam, **MASKGEN_KW)

    # Enumerate images named 0.png, 1.png, ...
    images = sorted([p for p in INPUT_DIR.glob("*.png")], key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not images:
        raise FileNotFoundError(f"No images found in {INPUT_DIR} (expected 0.png, 1.png, ...).")

    for img_path in images:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Could not read {img_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # --- Run SAM automatic mask generator
        masks = mask_generator.generate(img_rgb)  # list of dicts with 'segmentation', 'area', 'bbox', 'predicted_iou', ...

        # --- Save label PNG
        label = _masks_to_label_png(masks, h, w)  # uint16 label map
        label_path = OUTPUT_DIR / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(label_path), label.astype(np.uint16))  # preserves ids

        # --- Save overlay visualization
        overlay = _draw_contours_overlay(img_bgr, masks)
        overlay_path = OUTPUT_DIR / f"{img_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)

        # --- Save JSON with mask metadata (RLE, bbox, area, score)
        out_items = []
        for i, m in enumerate(masks, start=1):
            rle = _rle_encode_boolmask(m["segmentation"])
            x, y, bw, bh = m["bbox"]
            out_items.append({
                "id": i,
                "rle": rle,                       # COCO RLE (if pycocotools available) or simple fallback
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(m["area"]),
                "predicted_iou": float(m.get("predicted_iou", 0.0)),
                "stability_score": float(m.get("stability_score", 0.0)),
            })

        json_path = OUTPUT_DIR / f"{img_path.stem}_masks.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"image": img_path.name, "height": h, "width": w, "masks": out_items}, f, indent=2)

        print(f"[OK] {img_path.name} -> {label_path.name}, {overlay_path.name}, {json_path.name}")

if __name__ == "__main__":
    main()
