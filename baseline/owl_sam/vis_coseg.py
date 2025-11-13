#!/usr/bin/env python3
# run_vis_binary.py — visualize binary co-seg: blue=accepted, red=rejected

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------- hard-coded config ----------
INPUTS_DIR  = Path("inputs")
OUTPUTS_DIR = Path("outputs")
LABEL       = "engine"     # <-- change to your label slug
THRESH      = 0.95        # probs[1] >= THRESH => accepted
ALPHA_ACC   = 0.70       # overlay opacity for accepted (blue)
ALPHA_REJ   = 0.50       # overlay opacity for rejected (red)
DIFF_THRESH = 12         # must match encoder.py
# --------------------------------------

BLUE = (65, 105, 225)    # accepted
RED  = (220, 20, 60)     # rejected

def _make_faint(arr_rgb: np.ndarray) -> np.ndarray:
    arr = arr_rgb.astype(np.float32)
    white_mask = (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
    fade = arr * 0.2 + 255.0 * 0.8
    arr[~white_mask] = fade[~white_mask]
    return np.clip(arr, 0, 255).astype(np.uint8)

def _reconstruct_fragment_mask_full(frag_overlay_path: Path, original_image_path: Path, diff_thresh: int = DIFF_THRESH) -> np.ndarray:
    overlay = np.array(Image.open(frag_overlay_path).convert("RGB"))
    orig = np.array(Image.open(original_image_path).convert("RGB"))
    faint = _make_faint(orig)
    diff = np.abs(overlay.astype(np.int16) - faint.astype(np.int16))
    mask = (diff > diff_thresh).any(axis=2)
    return mask

def _alpha_blend(base: Image.Image, mask: np.ndarray, color, alpha: float) -> Image.Image:
    base = base.convert("RGBA")
    h, w = mask.shape
    overlay = Image.new("RGBA", (w, h), color + (0,))
    a = (mask.astype(np.uint8) * int(255 * alpha))
    overlay.putalpha(Image.fromarray(a, mode="L"))
    return Image.alpha_composite(base, overlay)

def _draw_tag(img: Image.Image, text: str, xy=(8, 8)) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.load_default()
    except:
        fnt = None
    pad = 4
    tw, th = d.textbbox((0, 0), text, font=fnt)[2:]
    d.rectangle([xy[0]-pad, xy[1]-pad, xy[0]+tw+pad, xy[1]+th+pad], fill=(255, 255, 255, 210))
    d.text(xy, text, fill=(0, 0, 0), font=fnt)
    return im

def main():
    x_dirs = [p for p in sorted(OUTPUTS_DIR.iterdir()) if p.is_dir()]
    if not x_dirs:
        print("[vis] No outputs/* folders found.")
        return

    for x_dir in x_dirs:
        x_id = x_dir.name
        sketch_path = INPUTS_DIR / f"{x_id}.png"
        frags_dir = x_dir / "fragments" / LABEL
        coseg_dir = x_dir / "co_seg"

        if not sketch_path.exists() or not frags_dir.exists() or not coseg_dir.exists():
            # skip images that don't have all required inputs
            continue

        # start from the plain sketch
        base = Image.open(sketch_path).convert("RGBA")

        # list all overlay fragments for this label
        overlay_paths = sorted(frags_dir.glob(f"{LABEL}_frag_*_overlay.png"))
        if not overlay_paths:
            continue

        accepted = rejected = 0
        out = base
        for overlay_path in overlay_paths:
            # parse frag index
            stem = overlay_path.stem  # e.g., wing_frag_3_overlay
            try:
                frag_idx = int(stem.split("_frag_")[1].split("_")[0])
            except Exception:
                frag_idx = -1

            assign_json = coseg_dir / f"{LABEL}_frag_{frag_idx}_assign.json"
            if not assign_json.exists():
                # no decision for this fragment -> skip
                continue

            with open(assign_json, "r") as f:
                payload = json.load(f)
            probs = payload.get("probs", None)
            if not probs or len(probs) < 2:
                continue
            p_true = float(probs[1])  # group-1 = "belongs"
            print("p_true", p_true)

            mask = _reconstruct_fragment_mask_full(overlay_path, sketch_path, diff_thresh=DIFF_THRESH)
            if mask.sum() == 0:
                continue

            if p_true >= THRESH:
                out = _alpha_blend(out, mask, BLUE, ALPHA_ACC)
                accepted += 1
            else:
                out = _alpha_blend(out, mask, RED, ALPHA_REJ)
                rejected += 1

        out = _draw_tag(out, f"x={x_id}  label={LABEL}  accepted={accepted}  rejected={rejected}")

        vis_dir = x_dir / "coseg_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        out_path = vis_dir / f"{LABEL}_overlay.png"
        out.save(out_path)
        print(f"[vis] x={x_id} → {out_path}")

if __name__ == "__main__":
    main()
