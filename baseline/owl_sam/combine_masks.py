#!/usr/bin/env python
# Combine masks from multiple ctrl_* runs by per-label union.
# Writes per-label union masks + overlay into owl_sam_output/final_output/.
# No prints.

import os, re, json
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import cv2

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def _read_mask(path: str) -> np.ndarray:
    # Returns boolean HxW
    m = np.array(Image.open(path).convert("L"))
    return (m > 127)

def _color_for(lbl: str):
    # BGR ints for cv2
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return int(c[2]), int(c[1]), int(c[0])

def combine_all(
    base_out_dir: str,          # e.g., "owl_sam_output"
    components_json: str,       # list of labels (supports {name,count} or plain strings)
    original_image_path: str,   # used for overlay size / visualization
):
    base = Path(base_out_dir)
    final_dir = base / "final_output"
    final_dir.mkdir(parents=True, exist_ok=True)

    # load labels
    payload = json.load(open(components_json, "r"))
    comps = payload.get("components", [])
    labels: List[str] = []
    if comps and isinstance(comps[0], dict):
        for it in comps:
            n = str(it.get("name", "")).strip()
            if n: labels.append(n)
    else:
        for n in comps:
            if isinstance(n, str) and n.strip():
                labels.append(n.strip())

    # find ctrl_* subfolders
    ctrl_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("ctrl_")])

    # determine target size (from first available mask; else original image)
    H = W = None
    for d in ctrl_dirs:
        for lbl in labels:
            mpath = next((p for p in d.glob(f"{_slug(lbl)}_mask.png")), None)
            if mpath and mpath.exists():
                m_test = _read_mask(str(mpath))
                H, W = m_test.shape
                break
        if H is not None:
            break
    if H is None or W is None:
        if os.path.exists(original_image_path):
            img = Image.open(original_image_path).convert("RGB")
            W, H = img.size
        else:
            # nothing to size against; bail gracefully with zero-sized output
            return

    # union per-label
    unions: Dict[str, np.ndarray] = {}
    for lbl in labels:
        slug = _slug(lbl)
        union_mask = np.zeros((H, W), dtype=bool)
        for d in ctrl_dirs:
            mpath = next((p for p in d.glob(f"{slug}_mask.png")), None)
            if mpath and mpath.exists():
                m = _read_mask(str(mpath))
                # if mask size mismatches, resize safely to (W,H)
                if m.shape != (H, W):
                    m = np.array(Image.fromarray((m.astype(np.uint8)*255)).resize((W, H), Image.NEAREST)) > 127
                union_mask |= m
        unions[lbl] = union_mask

    # overlay on original sketch (or blank)
    if os.path.exists(original_image_path):
        base_bgr = cv2.cvtColor(np.array(Image.open(original_image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        if base_bgr.shape[:2] != (H, W):
            base_bgr = cv2.resize(base_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        base_bgr = np.zeros((H, W, 3), np.uint8)

    overlay = base_bgr.astype(np.float32)
    alpha = 0.4
    for lbl, m in unions.items():
        if m.size == 0:
            continue
        col = np.array(_color_for(lbl), dtype=np.float32)  # (B,G,R)
        m3 = m[:, :, None].astype(np.float32)  # HxWx1
        overlay = overlay * (1.0 - alpha * m3) + (alpha * m3) * col  # safe broadcasting

    overlay_u8 = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.imwrite(str(final_dir / "overlay.png"), overlay_u8)

    # save per-label union masks
    for lbl, m in unions.items():
        outp = final_dir / f"{_slug(lbl)}_mask.png"
        cv2.imwrite(str(outp), (m.astype(np.uint8) * 255))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out_dir", required=True)
    ap.add_argument("--components_json", required=True)
    ap.add_argument("--original_image", required=True)
    args = ap.parse_args()
    combine_all(args.base_out_dir, args.components_json, args.original_image)
