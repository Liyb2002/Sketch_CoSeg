#!/usr/bin/env python
# Probabilistic mask aggregation + per-label overlays.
# For a given base_out_dir (e.g. outputs/1), expects:
#   base_out_dir/ctrl_*/{slug}_mask.png
# Writes to:
#   base_out_dir/probabilistic_aggregate/
#
# No prints; can be imported or run as a CLI.

import os
import re
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import cv2

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def _read_mask(path: str) -> np.ndarray:
    """
    Load a binary mask (HxW, bool) from a PNG.
    Any value > 127 is treated as True.
    """
    m = np.array(Image.open(path).convert("L"))
    return (m > 127)

def _color_for(lbl: str):
    """
    Deterministic per-label color (B,G,R) for OpenCV.
    """
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    # Return as BGR
    return int(c[2]), int(c[1]), int(c[0])

def probabilistic_combine(
    base_out_dir: str,          # e.g., "outputs/1"
    components_json: str,       # e.g., "inputs/components_1.json"
    original_image_path: str,   # e.g., "inputs/1.png"
):
    base = Path(base_out_dir)
    out_dir = base / "probabilistic_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- load labels from components_json --------
    payload = json.load(open(components_json, "r"))
    comps = payload.get("components", [])
    labels: List[str] = []

    if comps and isinstance(comps, list) and comps and isinstance(comps[0], dict):
        for it in comps:
            n = str(it.get("name", "")).strip()
            if n:
                labels.append(n)
    else:
        for n in comps:
            if isinstance(n, str) and n.strip():
                labels.append(n.strip())

    # nothing to do
    if not labels:
        return

    # -------- find ctrl_* dirs under base_out_dir --------
    ctrl_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("ctrl_")]
    )
    num_ctrl = len(ctrl_dirs)
    if num_ctrl == 0:
        return

    # -------- determine target size (H, W) --------
    H = W = None
    for d in ctrl_dirs:
        for lbl in labels:
            slug = _slug(lbl)
            mpath = next(d.glob(f"{slug}_mask.png"), None)
            if mpath and mpath.exists():
                m_test = _read_mask(str(mpath))
                if m_test.ndim == 2:
                    H, W = m_test.shape
                    break
        if H is not None:
            break

    # fallback: use original image size if no mask found
    if H is None or W is None:
        if os.path.exists(original_image_path):
            img = Image.open(original_image_path).convert("RGB")
            W, H = img.size
        else:
            return

    # -------- base image for overlays --------
    if os.path.exists(original_image_path):
        base_bgr = cv2.cvtColor(
            np.array(Image.open(original_image_path).convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
        if base_bgr.shape[:2] != (H, W):
            base_bgr = cv2.resize(base_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        base_bgr = np.zeros((H, W, 3), np.uint8)

    base_float = base_bgr.astype(np.float32)

    # -------- accumulate per-label counts --------
    counts: Dict[str, np.ndarray] = {
        lbl: np.zeros((H, W), dtype=np.float32) for lbl in labels
    }

    for d in ctrl_dirs:
        for lbl in labels:
            slug = _slug(lbl)
            mpath = next(d.glob(f"{slug}_mask.png"), None)
            if not (mpath and mpath.exists()):
                # No mask from this ctrl for this label: contributes 0
                continue

            m = _read_mask(str(mpath))

            # resize mask if needed
            if m.shape != (H, W):
                m_img = Image.fromarray((m.astype(np.uint8) * 255))
                m_resized = m_img.resize((W, H), Image.NEAREST)
                m = np.array(m_resized) > 127

            counts[lbl] += m.astype(np.float32)

    # -------- build probability maps & per-label overlays --------
    alpha_max = 0.6  # max blend strength
    prob_maps: Dict[str, np.ndarray] = {}

    for lbl in labels:
        c = counts[lbl]
        if num_ctrl > 0:
            prob = c / float(num_ctrl)
        else:
            prob = np.zeros_like(c, dtype=np.float32)

        prob = np.clip(prob, 0.0, 1.0)
        prob_maps[lbl] = prob

        # save raw probability map as grayscale
        prob_u8 = (prob * 255.0).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{_slug(lbl)}_prob.png"), prob_u8)

        # per-label overlay: blend base image toward label color based on prob
        col = np.array(_color_for(lbl), dtype=np.float32)  # BGR
        m = prob[:, :, None]  # HxWx1
        overlay = base_float * (1.0 - alpha_max * m) + col * (alpha_max * m)
        overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{_slug(lbl)}_overlay.png"), overlay_u8)

    # -------- global overlay across all labels --------
    global_overlay = base_float.copy()
    for lbl, prob in prob_maps.items():
        if prob.size == 0:
            continue
        col = np.array(_color_for(lbl), dtype=np.float32)
        m = prob[:, :, None]
        global_overlay = global_overlay * (1.0 - alpha_max * m) + col * (alpha_max * m)

    global_u8 = np.clip(global_overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "all_labels_overlay.png"), global_u8)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out_dir", required=True, help="e.g. outputs/1")
    ap.add_argument("--components_json", required=True, help="e.g. inputs/components_1.json")
    ap.add_argument("--original_image", required=True, help="e.g. inputs/1.png")
    args = ap.parse_args()

    probabilistic_combine(
        base_out_dir=args.base_out_dir,
        components_json=args.components_json,
        original_image_path=args.original_image,
    )

# python probabilistic_combine.py \
#   --base_out_dir outputs/1 \
#   --components_json inputs/components_1.json \
#   --original_image inputs/1.png