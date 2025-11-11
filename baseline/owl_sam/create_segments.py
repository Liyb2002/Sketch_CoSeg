#!/usr/bin/env python
# Create per-label fragments directly from multiple ctrl_* masks.
#
# Given:
#   base_out_dir = outputs/x
#   components_json = inputs/components_x.json
#   original_image = inputs/x.png
#
# Expects:
#   base_out_dir/ctrl_*/{slug}_mask.png
#
# For each label:
#   - Collect all masks across ctrl_* dirs.
#   - Encode which masks cover each pixel as a bit pattern.
#   - For each non-zero pattern, run connected components.
#   - Each connected component is a fragment.
#
# For each fragment we save:
#   outputs/x/fragments/{slug}/{slug}_frag_{k}_mask.png
#       (binary, cropped to fragment bbox)
#   outputs/x/fragments/{slug}/{slug}_frag_{k}_overlay.png
#       (FULL-SIZE image: faint sketch + strong label-colored fragment)
#
# For each label we also save a mosaic:
#   outputs/x/probabilistic_aggregate/{slug}_fragments_overlay.png
#       (FULL-SIZE: faint sketch + all fragments in randomized colors)
#
# No prints. Safe to import.

import os
import re
import json
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import cv2

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def _load_labels(components_json: str) -> List[str]:
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

    # dedupe keep order
    seen = set()
    uniq = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq

def _read_mask_bool(path: Path) -> np.ndarray:
    """Load binary mask as bool HxW (True where mask > 127)."""
    m = np.array(Image.open(path).convert("L"))
    return (m > 127)

def _label_color(lbl: str) -> np.ndarray:
    """
    Deterministic strong color per label for single-fragment overlays.
    Returns (B, G, R) float32.
    """
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(80, 255, size=3, dtype=np.int32)
    return c.astype(np.float32)  # B,G,R

def _fragment_color(lbl: str, frag_idx: int) -> np.ndarray:
    """
    Deterministic random color per fragment (per label) for mosaic.
    Returns (B, G, R) float32.
    """
    seed = abs(hash((lbl, frag_idx))) % (2**32)
    rng = np.random.default_rng(seed)
    c = rng.integers(80, 255, size=3, dtype=np.int32)
    return c.astype(np.float32)

def _make_faint(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Make sketch faint:
      - near-white stays white
      - darker strokes pushed toward white.
    """
    arr = arr_rgb.astype(np.float32)
    white_mask = (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
    fade = arr * 0.2 + 255.0 * 0.8
    arr[~white_mask] = fade[~white_mask]
    return np.clip(arr, 0, 255).astype(np.uint8)

def create_segments(
    base_out_dir: str,        # e.g. "outputs/1"
    components_json: str,     # e.g. "inputs/components_1.json"
    original_image_path: str, # e.g. "inputs/1.png"
    min_area: int = 32,
    connectivity: int = 8,    # 4 or 8
):
    base = Path(base_out_dir)
    if not base.exists():
        return

    if not os.path.exists(original_image_path):
        return

    labels = _load_labels(components_json)
    if not labels:
        return

    # Original sketch as RGB + faint version
    orig = Image.open(original_image_path).convert("RGB")
    orig_np = np.array(orig)
    H0, W0 = orig_np.shape[:2]

    faint_full = _make_faint(orig_np).astype(np.float32)

    # Find ctrl_* dirs
    ctrl_dirs = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("ctrl_")]
    )
    if not ctrl_dirs:
        return

    # Ensure probabilistic_aggregate dir for mosaics
    prob_dir = base / "probabilistic_aggregate"
    prob_dir.mkdir(parents=True, exist_ok=True)

    for lbl in labels:
        slug = _slug(lbl)

        # ---- collect all masks for this label ----
        mask_list: List[np.ndarray] = []
        for d in ctrl_dirs:
            mpath = d / f"{slug}_mask.png"
            if mpath.exists():
                m = _read_mask_bool(mpath)
                if m.shape != (H0, W0):
                    m_img = Image.fromarray((m.astype(np.uint8) * 255))
                    m_img = m_img.resize((W0, H0), Image.NEAREST)
                    m = (np.array(m_img) > 127)
                mask_list.append(m)

        if not mask_list:
            continue

        masks = np.stack(mask_list, axis=0)  # [N, H, W]
        n_masks = masks.shape[0]
        if n_masks > 32:
            masks = masks[:32]
            n_masks = 32

        # ---- per-pixel coverage pattern as bitset ----
        pattern = np.zeros((H0, W0), dtype=np.uint32)
        for k in range(n_masks):
            pattern |= (masks[k].astype(np.uint32) << k)

        uniq_vals = np.unique(pattern)
        uniq_vals = uniq_vals[uniq_vals != 0]
        if uniq_vals.size == 0:
            continue

        # dirs
        frag_dir = base / "fragments" / slug
        frag_dir.mkdir(parents=True, exist_ok=True)

        # mosaic for this label: faint sketch + each fragment random color
        mosaic = faint_full.copy()

        # store fragment masks for mosaic coloring (full-size)
        fragment_masks_full: List[np.ndarray] = []

        frag_idx = 0

        for val in uniq_vals:
            bin_mask = (pattern == val).astype(np.uint8)
            if bin_mask.sum() < min_area:
                continue

            num_cc, cc_map = cv2.connectedComponents(bin_mask, connectivity=connectivity)
            for cc_id in range(1, num_cc):
                comp_mask = (cc_map == cc_id)
                area = int(comp_mask.sum())
                if area < min_area:
                    continue

                ys, xs = np.where(comp_mask)
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()

                comp_crop = comp_mask[y1:y2+1, x1:x2+1]
                if comp_crop.max() == 0:
                    continue

                # 1) Save cropped binary mask
                mask_img = (comp_crop.astype(np.uint8) * 255)
                mask_path = frag_dir / f"{slug}_frag_{frag_idx}_mask.png"
                Image.fromarray(mask_img).save(mask_path)

                # 2) Full-size overlay for THIS fragment:
                #    faint full sketch + strong label color only on this fragment.
                label_col_bgr = _label_color(lbl)
                label_col_rgb = np.array(
                    [label_col_bgr[2], label_col_bgr[1], label_col_bgr[0]], dtype=np.float32
                )
                full_overlay = faint_full.copy()
                mask_bool_full = comp_mask  # HxW

                alpha_label = 0.9
                full_overlay[mask_bool_full] = (
                    alpha_label * label_col_rgb
                    + (1.0 - alpha_label) * faint_full[mask_bool_full]
                )

                overlay_path = frag_dir / f"{slug}_frag_{frag_idx}_overlay.png"
                Image.fromarray(
                    np.clip(full_overlay, 0, 255).astype(np.uint8)
                ).save(overlay_path)

                # Save this full-size fragment mask for mosaic coloring
                fragment_masks_full.append(comp_mask.copy())
                frag_idx += 1

        # 3) Build per-label mosaic with random color per fragment
        if frag_idx > 0 and fragment_masks_full:
            mosaic = faint_full.copy()
            for i, fm in enumerate(fragment_masks_full):
                col_bgr = _fragment_color(lbl, i)
                col_rgb = np.array(
                    [col_bgr[2], col_bgr[1], col_bgr[0]], dtype=np.float32
                )
                alpha_frag = 0.9
                mask_bool = fm.astype(bool)
                mosaic[mask_bool] = (
                    alpha_frag * col_rgb + (1.0 - alpha_frag) * mosaic[mask_bool]
                )

            mosaic_path = prob_dir / f"{slug}_fragments_overlay.png"
            Image.fromarray(
                np.clip(mosaic, 0, 255).astype(np.uint8)
            ).save(mosaic_path)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out_dir", required=True, help="e.g. outputs/1")
    ap.add_argument("--components_json", required=True, help="e.g. inputs/components_1.json")
    ap.add_argument("--original_image", required=True, help="e.g. inputs/1.png")
    ap.add_argument("--min_area", type=int, default=32)
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    args = ap.parse_args()

    create_segments(
        base_out_dir=args.base_out_dir,
        components_json=args.components_json,
        original_image_path=args.original_image,
        min_area=args.min_area,
        connectivity=args.connectivity,
    )

if __name__ == "__main__":
    main()
