#!/usr/bin/env python
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image

# ----------------- config -----------------
INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")
DIFF_THRESH = 12  # threshold to detect fragment from overlay


# ----------------- helpers -----------------
def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _make_faint(arr_rgb: np.ndarray) -> np.ndarray:
    """Recreate faint background used for overlays."""
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


def _sample_points_from_mask(mask: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask)
    n = xs.size
    if n == 0:
        return []
    if n <= k:
        return [(int(xs[i]), int(ys[i])) for i in range(n)]
    idxs = np.linspace(0, n - 1, k, dtype=int)
    return [(int(xs[i]), int(ys[i])) for i in idxs]


def _load_owl_boxes(owl_json_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load OWL boxes from either `owl_results.json` or `owl_boxes.json`."""
    if not owl_json_path.exists():
        return {}

    with open(owl_json_path, "r") as f:
        data = json.load(f)

    if "labels" in data and isinstance(data["labels"], dict):
        labels_dict = data["labels"]
    else:
        labels_dict = data  # your format

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lbl, v in labels_dict.items():
        if not isinstance(v, dict):
            continue
        boxes = np.array(v.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        scores = np.array(v.get("scores", []), dtype=np.float32).reshape(-1)
        out[lbl] = {"boxes": boxes, "scores": scores}
    return out


def _points_inside_box(points: List[Tuple[int, int]], box: Tuple[float, float, float, float]) -> Tuple[int, List[bool]]:
    """Return (count_inside, [bool per point]) for inside/on-border."""
    x1, y1, x2, y2 = box
    flags = [(x1 <= px <= x2 and y1 <= py <= y2) for (px, py) in points]
    return sum(flags), flags


def _determine_reference_label() -> Tuple[str, str]:
    """From outputs/0/fragments: find first label and its human-readable text."""
    x0_frag_root = OUTPUTS_DIR / "0" / "fragments"
    if not x0_frag_root.exists():
        raise SystemExit("[ERR] outputs/0/fragments not found.")

    label_slugs = sorted([p.name for p in x0_frag_root.iterdir() if p.is_dir()])
    if not label_slugs:
        raise SystemExit("[ERR] No label subfolders in outputs/0/fragments.")

    ref_label_slug = label_slugs[0]
    ref_label_text = ref_label_slug

    comp0_path = INPUTS_DIR / "components_0.json"
    if comp0_path.exists():
        with open(comp0_path, "r") as f:
            payload0 = json.load(f)
        comps0 = payload0.get("components", [])
        if comps0 and isinstance(comps0[0], dict):
            for it in comps0:
                n = str(it.get("name", "")).strip()
                if _slug(n) == ref_label_slug:
                    ref_label_text = n
                    break
        else:
            for n in comps0:
                if isinstance(n, str) and _slug(n) == ref_label_slug:
                    ref_label_text = n.strip()
                    break

    print(f"[INFO] Reference label: slug='{ref_label_slug}', text='{ref_label_text}'")
    return ref_label_slug, ref_label_text


# ----------------- main -----------------
def main():
    ref_label_slug, ref_label_text = _determine_reference_label()

    for x_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not x_dir.is_dir():
            continue
        x_id = x_dir.name

        sketch_path = INPUTS_DIR / f"{x_id}.png"
        fragments_dir = x_dir / "fragments" / ref_label_slug

        if not sketch_path.exists() or not fragments_dir.exists():
            continue

        overlay_paths = sorted(fragments_dir.glob(f"{ref_label_slug}_frag_*_overlay.png"))
        if not overlay_paths:
            continue

        for overlay_path in overlay_paths:
            frag_idx = overlay_path.stem
            try:
                frag_idx = int(frag_idx.split("_frag_")[1].split("_")[0])
            except Exception:
                pass

            frag_mask = _reconstruct_fragment_mask_full(overlay_path, sketch_path)
            if frag_mask.sum() == 0:
                continue

            sample_points = _sample_points_from_mask(frag_mask, k=3)
            if not sample_points:
                continue

            contrib_ctrls = []

            for ctrl_dir in sorted(x_dir.glob("ctrl_*")):
                if not ctrl_dir.is_dir():
                    continue

                owl_path = ctrl_dir / "owl_results.json"
                if not owl_path.exists():
                    owl_path = ctrl_dir / "owl_boxes.json"
                if not owl_path.exists():
                    continue

                boxes_data = _load_owl_boxes(owl_path)
                if not boxes_data:
                    continue

                # collect boxes for our label
                keys = [ref_label_text, ref_label_slug, ref_label_text.lower(), ref_label_slug.lower()]
                all_boxes = []
                for k in keys:
                    if k in boxes_data:
                        b = boxes_data[k]["boxes"]
                        if b.size > 0:
                            all_boxes.append(b)
                if not all_boxes:
                    continue
                all_boxes = np.concatenate(all_boxes, axis=0)

                for b in all_boxes:
                    x1, y1, x2, y2 = map(float, b.tolist())
                    inside_count, _ = _points_inside_box(sample_points, (x1, y1, x2, y2))
                    if inside_count == len(sample_points):
                        contrib_ctrls.append(ctrl_dir.name)
                        break

            if contrib_ctrls:
                print(f"[x={x_id}] fragment {frag_idx} -> contributes: {sorted(set(contrib_ctrls))}")


if __name__ == "__main__":
    main()
