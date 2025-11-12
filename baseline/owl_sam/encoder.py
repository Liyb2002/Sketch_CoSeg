#!/usr/bin/env python
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

# ----------------- config -----------------
INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")
DIFF_THRESH = 12  # threshold to detect fragment from overlay
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(0)  # reproducible ctrl_* choice


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


def _reconstruct_fragment_mask_full(
    frag_overlay_path: Path,
    original_image_path: Path,
    diff_thresh: int = DIFF_THRESH,
) -> np.ndarray:
    """Generate HxW bool mask of fragment pixels from overlay."""
    overlay = np.array(Image.open(frag_overlay_path).convert("RGB"))
    orig = np.array(Image.open(original_image_path).convert("RGB"))
    faint = _make_faint(orig)
    diff = np.abs(overlay.astype(np.int16) - faint.astype(np.int16))
    mask = (diff > diff_thresh).any(axis=2)
    return mask


def _sample_points_from_mask(mask: np.ndarray, k: int = 3) -> List[Tuple[int, int]]:
    """Deterministically sample up to k (x, y) points from True pixels."""
    ys, xs = np.where(mask)
    n = xs.size
    if n == 0:
        return []
    if n <= k:
        return [(int(xs[i]), int(ys[i])) for i in range(n)]
    idxs = np.linspace(0, n - 1, k, dtype=int)
    return [(int(xs[i]), int(ys[i])) for i in idxs]


def _load_owl_boxes(owl_json_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load OWL boxes; supports:
      {
        "wheel": { "boxes": [...], "scores": [...], ... }, ...
      }
    or:
      {
        "labels": { "wheel": { "boxes": [...], "scores": [...] }, ... }
      }
    """
    if not owl_json_path.exists():
        return {}

    with open(owl_json_path, "r") as f:
        data = json.load(f)

    if "labels" in data and isinstance(data["labels"], dict):
        labels_dict = data["labels"]
    else:
        labels_dict = data

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for lbl, v in labels_dict.items():
        if not isinstance(v, dict):
            continue
        boxes = np.array(v.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        scores = np.array(v.get("scores", []), dtype=np.float32).reshape(-1)
        out[lbl] = {"boxes": boxes, "scores": scores}
    return out


def _points_inside_box(
    points: List[Tuple[int, int]],
    box: Tuple[float, float, float, float],
) -> bool:
    """True iff ALL points are inside/on the box."""
    x1, y1, x2, y2 = box
    for px, py in points:
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            return False
    return True


def _determine_reference_label() -> Tuple[str, str]:
    """
    From outputs/0/fragments:
      - take first subfolder -> ref_label_slug
      - map to human-readable text via inputs/components_0.json if possible
    """
    x0_frag_root = OUTPUTS_DIR / "0" / "fragments"
    if not x0_frag_root.exists():
        raise SystemExit("[ERR] outputs/0/fragments not found.")

    label_slugs = sorted(p.name for p in x0_frag_root.iterdir() if p.is_dir())
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


def _get_object_name(x_id: str) -> str:
    """Read object_name from components_{x}.json, default 'object'."""
    comp_path = INPUTS_DIR / f"components_{x_id}.json"
    if not comp_path.exists():
        return "object"
    with open(comp_path, "r") as f:
        payload = json.load(f)
    name = str(payload.get("object_name", "object")).strip()
    return name or "object"


# ---- new: ensure minimum image size to avoid 1-pixel dims ambiguity ----
def _ensure_min_size(img: Image.Image, min_size: int = 2) -> Image.Image:
    """
    Ensure the image is at least min_size x min_size by padding on a white canvas.
    This avoids shape ambiguities like (1, W, 3) that confuse preprocessors.
    """
    img = img.convert("RGB")
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    new_w, new_h = max(w, min_size), max(h, min_size)
    out = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    out.paste(img, (0, 0))
    return out


def _mask_ctrl_fragment_image(
    ctrl_img: np.ndarray,
    frag_mask: np.ndarray,
) -> Image.Image:
    """
    White background, keep ctrl_img where frag_mask is True.
    Then crop to tight bbox of mask.
    """
    H, W = frag_mask.shape

    # If sizes differ, resize mask to ctrl size (binary NN)
    if (ctrl_img.shape[0], ctrl_img.shape[1]) != (H, W):
        Hc, Wc = ctrl_img.shape[:2]
        frag_mask = np.array(
            Image.fromarray((frag_mask.astype(np.uint8) * 255)).resize(
                (Wc, Hc), resample=Image.NEAREST
            )
        ) > 0
        H, W = Hc, Wc

    masked = np.ones_like(ctrl_img, dtype=np.uint8) * 255
    masked[frag_mask] = ctrl_img[frag_mask]

    ys, xs = np.where(frag_mask)
    if ys.size == 0 or xs.size == 0:
        return _ensure_min_size(Image.fromarray(masked))

    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = masked[y1:y2, x1:x2, :]
    return _ensure_min_size(Image.fromarray(crop))


def _crop_box_image(ctrl_img: np.ndarray, box: Tuple[float, float, float, float]) -> Image.Image:
    """Crop ctrl_img to the given (x1, y1, x2, y2) box (clamped)."""
    x1, y1, x2, y2 = box
    h, w = ctrl_img.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return _ensure_min_size(Image.fromarray(ctrl_img))
    return _ensure_min_size(Image.fromarray(ctrl_img[y1:y2, x1:x2, :]))


# ----------------- CLIP helpers -----------------
def _prepare_clip():
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def _encode_image(model, processor, img: Image.Image) -> np.ndarray:
    # Ensure valid size & explicit channels-last format to avoid ambiguity.
    img = _ensure_min_size(img.convert("RGB"), min_size=2)
    inputs = processor(
        images=img,
        return_tensors="pt",
        input_data_format="channels_last"  # disambiguate degenerate shapes
    ).to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    v = feats[0].detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


def _encode_text(model, processor, text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    v = feats[0].detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


# ----------------- main -----------------
def main():
    ref_label_slug, ref_label_text = _determine_reference_label()
    model, processor = _prepare_clip()

    for x_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not x_dir.is_dir():
            continue
        x_id = x_dir.name

        sketch_path = INPUTS_DIR / f"{x_id}.png"
        fragments_dir = x_dir / "fragments" / ref_label_slug
        realism_dir = x_dir / "realism_imgs"
        if not sketch_path.exists() or not fragments_dir.exists() or not realism_dir.exists():
            continue

        overlay_paths = sorted(fragments_dir.glob(f"{ref_label_slug}_frag_*_overlay.png"))
        if not overlay_paths:
            continue

        # embeddings output folder
        emb_dir = x_dir / "embeddings" / ref_label_slug
        emb_dir.mkdir(parents=True, exist_ok=True)

        object_name = _get_object_name(x_id)
        text_for_x = f"{ref_label_text} of a {object_name}"

        for overlay_path in overlay_paths:
            # parse fragment index
            stem = overlay_path.stem  # {slug}_frag_k_overlay
            try:
                frag_idx_part = stem.split("_frag_")[1]
                frag_idx = int(frag_idx_part.split("_")[0])
            except Exception:
                frag_idx = -1

            # fragment mask
            frag_mask = _reconstruct_fragment_mask_full(overlay_path, sketch_path)
            if frag_mask.sum() == 0:
                continue

            # sample points
            sample_points = _sample_points_from_mask(frag_mask, k=3)
            if not sample_points:
                continue

            # find ALL contributing ctrl_* and remember the matching box
            contrib: Dict[str, Tuple[float, float, float, float]] = {}

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
                    if _points_inside_box(sample_points, (x1, y1, x2, y2)):
                        contrib[ctrl_dir.name] = (x1, y1, x2, y2)
                        break  # one good box per ctrl is enough

            if not contrib:
                # no ctrl supports this fragment → skip embedding
                continue

            # pick one ctrl_* at random and get its box
            ctrl_name = random.choice(list(contrib.keys()))
            box = contrib[ctrl_name]

            # load ctrl image
            ctrl_img_path = realism_dir / f"{ctrl_name}.png"
            if not ctrl_img_path.exists():
                continue
            ctrl_img = np.array(Image.open(ctrl_img_path).convert("RGB"))

            # 1) emb_mask
            img_mask = _mask_ctrl_fragment_image(ctrl_img, frag_mask)
            emb_mask = _encode_image(model, processor, img_mask)

            # 2) emb_box
            img_box = _crop_box_image(ctrl_img, box)
            emb_box = _encode_image(model, processor, img_box)

            # 3) emb_text
            emb_text = _encode_text(model, processor, text_for_x)

            # save npz
            out_path = emb_dir / f"{ref_label_slug}_frag_{frag_idx}_embed.npz"
            meta = {
                "x_id": x_id,
                "label": ref_label_text,
                "label_slug": ref_label_slug,
                "frag_index": frag_idx,
                "ctrl_name": ctrl_name,
                "box": [int(v) for v in box],
                "text": text_for_x,
            }
            np.savez_compressed(
                out_path,
                emb_mask=emb_mask,
                emb_box=emb_box,
                emb_text=emb_text,
                meta=json.dumps(meta),
            )

            print(f"[x={x_id}] frag {frag_idx}: saved -> {out_path.name}")

if __name__ == "__main__":
    main()
