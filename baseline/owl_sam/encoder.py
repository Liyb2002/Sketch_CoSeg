#!/usr/bin/env python
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import gc  # <--- NEW
import shutil

import torch
from transformers import CLIPProcessor, CLIPModel

# ----------------- config -----------------
INPUTS_DIR = Path("inputs")
OUTPUTS_DIR = Path("outputs")
DIFF_THRESH = 12  # threshold to detect fragment from overlay
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAG_GC_INTERVAL = 20  # <--- NEW: how many fragments before clearing CUDA cache

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


# ---------- map slugs to display text & enumerate all labels ----------
def _label_slug_to_text_map(comp0_path: Path) -> Dict[str, str]:
    """
    Build a mapping from slug -> human-readable text from inputs/components_0.json.
    Falls back to slug itself when no mapping is found.
    """
    slug2text: Dict[str, str] = {}
    if not comp0_path.exists():
        return slug2text

    with open(comp0_path, "r") as f:
        payload0 = json.load(f)

    comps0 = payload0.get("components", [])
    # supports: [{"name": "..."} ...] OR ["...", ...]
    if comps0 and isinstance(comps0[0], dict):
        for it in comps0:
            n = str(it.get("name", "")).strip()
            if n:
                slug2text[_slug(n)] = n
    else:
        for n in comps0:
            if isinstance(n, str):
                n = n.strip()
                if n:
                    slug2text[_slug(n)] = n
    return slug2text


def _determine_reference_labels() -> List[Tuple[str, str]]:
    """
    Read all label subfolders from outputs/0/fragments and map each slug
    to human-readable text using inputs/components_0.json when possible.
    Returns a sorted list of (slug, text).
    """
    x0_frag_root = OUTPUTS_DIR / "0" / "fragments"
    if not x0_frag_root.exists():
        raise SystemExit("[ERR] outputs/0/fragments not found.")

    label_slugs = sorted(p.name for p in x0_frag_root.iterdir() if p.is_dir())
    if not label_slugs:
        raise SystemExit("[ERR] No label subfolders in outputs/0/fragments.")

    slug2text = _label_slug_to_text_map(INPUTS_DIR / "components_0.json")

    out = []
    for slug in label_slugs:
        text = slug2text.get(slug, slug)
        out.append((slug, text))

    print("[INFO] Reference labels:")
    for slug, text in out:
        print(f"   - slug='{slug}', text='{text}'")
    return out
# ---------------------------------------------------------------------------


def _get_object_name(x_id: str) -> str:
    """Read object_name from components_{x}.json, default 'object'."""
    comp_path = INPUTS_DIR / f"components_{x_id}.json"
    if not comp_path.exists():
        return "object"
    with open(comp_path, "r") as f:
        payload = json.load(f)
    name = str(payload.get("object_name", "object")).strip()
    return name or "object"


# ---- ensure minimum image size to avoid 1-pixel dims ambiguity ----
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


# ----------------- CLIP helpers (text only) -----------------
def _prepare_clip():
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


def _encode_text(model, processor, text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    v = feats[0].detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


# ----------------- DINOv2 helpers (image embeddings) -----------------
def _prepare_dinov2():
    """
    Load DINOv2 ViT-B/14 and its preprocessing transform.
    Requires: torch.hub to access facebookresearch/dinov2.
    """
    import torchvision.transforms as T

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.Resize(256, interpolation=Image.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return model, transform


def _encode_image_dino(model, transform, img: Image.Image) -> np.ndarray:
    """
    Encode an image into a DINOv2 embedding.
    """
    img = _ensure_min_size(img.convert("RGB"), min_size=2)
    x = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
    with torch.no_grad():
        feats = model(x)  # (1, D) for DINOv2 hub models
    v = feats[0].detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(v) + 1e-8
    return v / n


# ----------------- main -----------------
def main():
    ref_labels = _determine_reference_labels()  # list[(slug, text)]

    # CLIP for text embeddings
    clip_model, clip_processor = _prepare_clip()

    # DINOv2 for image embeddings
    dino_model, dino_transform = _prepare_dinov2()

    frags_since_gc = 0  # fragment counter for CUDA cache flushing

    for x_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not x_dir.is_dir():
            continue
        x_id = x_dir.name

        # ---------------- CLEAN OUTPUT DIRS FOR THIS x_id ----------------
        embedded_items_dir = x_dir / "embedded_items"
        embeddings_root = x_dir / "embeddings"

        if embedded_items_dir.exists():
            shutil.rmtree(embedded_items_dir)
        if embeddings_root.exists():
            shutil.rmtree(embeddings_root)

        embedded_items_dir.mkdir(parents=True, exist_ok=True)
        embeddings_root.mkdir(parents=True, exist_ok=True)
        # -----------------------------------------------------------------

        sketch_path = INPUTS_DIR / f"{x_id}.png"
        if not sketch_path.exists():
            continue
        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))

        realism_dir = x_dir / "realism_imgs"
        if not realism_dir.exists():
            continue

        object_name = _get_object_name(x_id)

        # all ctrl_* dirs for this x_id
        ctrl_dirs = [d for d in sorted(x_dir.glob("ctrl_*")) if d.is_dir()]
        if not ctrl_dirs:
            continue

        for ref_label_slug, ref_label_text in ref_labels:
            fragments_dir = x_dir / "fragments" / ref_label_slug
            if not fragments_dir.exists():
                # label not present for this x_id → skip
                continue

            overlay_paths = sorted(
                fragments_dir.glob(f"{ref_label_slug}_frag_*_overlay.png")
            )
            if not overlay_paths:
                continue

            # embeddings output folder for this label
            emb_dir = embeddings_root / ref_label_slug
            emb_dir.mkdir(parents=True, exist_ok=True)

            text_for_x = f"{ref_label_text} of a {object_name}"
            # precompute text embedding once per x/label
            emb_text = _encode_text(clip_model, clip_processor, text_for_x)

            for overlay_path in overlay_paths:
                # parse fragment index: {slug}_frag_k_overlay.png
                stem = overlay_path.stem
                try:
                    frag_idx_part = stem.split("_frag_")[1]
                    frag_idx = int(frag_idx_part.split("_")[0])
                except Exception:
                    frag_idx = -1

                # fragment mask (in sketch space)
                frag_mask = _reconstruct_fragment_mask_full(overlay_path, sketch_path)
                if frag_mask.sum() == 0:
                    continue

                # sample points used to check box/OWL alignment
                sample_points = _sample_points_from_mask(frag_mask, k=3)
                if not sample_points:
                    continue

                # -------------------------------------------------
                # 1) Scan all ctrl_*: which ones choose this fragment?
                # -------------------------------------------------
                ctrl_infos = []  # {ctrl_name, matched, box}
                for ctrl_dir in ctrl_dirs:
                    ctrl_name = ctrl_dir.name

                    owl_path = ctrl_dir / "owl_results.json"
                    if not owl_path.exists():
                        owl_path = ctrl_dir / "owl_boxes.json"

                    boxes_data = _load_owl_boxes(owl_path) if owl_path.exists() else {}

                    matched = False
                    box_for_frag = None

                    if boxes_data:
                        keys = [
                            ref_label_text,
                            ref_label_slug,
                            ref_label_text.lower(),
                            ref_label_slug.lower(),
                        ]
                        all_boxes = []
                        for k in keys:
                            if k in boxes_data:
                                b = boxes_data[k]["boxes"]
                                if b.size > 0:
                                    all_boxes.append(b)
                        if all_boxes:
                            all_boxes = np.concatenate(all_boxes, axis=0)
                            for b in all_boxes:
                                x1, y1, x2, y2 = map(float, b.tolist())
                                if _points_inside_box(sample_points, (x1, y1, x2, y2)):
                                    matched = True
                                    box_for_frag = (x1, y1, x2, y2)
                                    break

                    ctrl_infos.append(
                        {
                            "ctrl_name": ctrl_name,
                            "matched": matched,
                            "box": box_for_frag,
                        }
                    )

                if not ctrl_infos:
                    continue

                chosen_infos = [ci for ci in ctrl_infos if ci["matched"]]
                unchosen_infos = [ci for ci in ctrl_infos if not ci["matched"]]

                # -------------------------------------------------
                # 2) Select 3 ctrl images for overlays:
                #    - ensure at least 1 chosen if possible
                #    - remaining 2 are random from all ctrl's
                # -------------------------------------------------
                if chosen_infos:
                    # pick 1 chosen
                    first = random.choice(chosen_infos)
                    # pick 2 more random from the rest (can be chosen or unchosen)
                    remaining = [ci for ci in ctrl_infos if ci is not first]
                    selected_infos = [first] + random.sample(remaining, 2)
                else:
                    # no chosen at all → just pick 3 random
                    selected_infos = random.sample(ctrl_infos, 3)

                # -------------------------------------------------
                # 3) Determine reference bounding box:
                #    - from a chosen image if any exist
                #    - else: fall back to full-image box later
                # -------------------------------------------------
                ref_box = None
                ref_box_source = None

                if chosen_infos:
                    chosen_with_box = [ci for ci in chosen_infos if ci["box"] is not None]
                    if chosen_with_box:
                        ci_ref = random.choice(chosen_with_box)
                        ref_box = ci_ref["box"]
                        ref_box_source = ci_ref["ctrl_name"]

                # -------------------------------------------------
                # 4) For each selected ctrl, compute mask+box embeddings
                # -------------------------------------------------
                for ci in selected_infos:
                    ctrl_name = ci["ctrl_name"]

                    ctrl_img_path = realism_dir / f"{ctrl_name}.png"
                    if not ctrl_img_path.exists():
                        continue
                    ctrl_img = np.array(Image.open(ctrl_img_path).convert("RGB"))

                    # mask embedding on THIS ctrl image (always)
                    img_mask = _mask_ctrl_fragment_image(ctrl_img, frag_mask)
                    emb_mask = _encode_image_dino(dino_model, dino_transform, img_mask)

                    # box embedding:
                    #  - if ref_box exists (from any chosen), use it
                    #  - otherwise (no chosen anywhere), full-image box
                    if ref_box is not None:
                        box = ref_box
                        matched_box = True
                    else:
                        h, w = ctrl_img.shape[:2]
                        box = (0.0, 0.0, float(w), float(h))
                        matched_box = False
                        ref_box_source = None

                    img_box = _crop_box_image(ctrl_img, box)
                    emb_box = _encode_image_dino(dino_model, dino_transform, img_box)

                    meta = {
                        "x_id": x_id,
                        "label": ref_label_text,
                        "label_slug": ref_label_slug,
                        "frag_index": frag_idx,
                        "ctrl_name": ctrl_name,
                        "box": [int(v) for v in box],
                        "text": text_for_x,          # plain text for emb_text
                        "matched_box": matched_box,  # True iff box came from OWL
                        "ref_box_source": ref_box_source,
                    }

                    out_path = (
                        emb_dir
                        / f"{ref_label_slug}_frag_{frag_idx}_{ctrl_name}_embed.npz"
                    )

                    np.savez_compressed(
                        out_path,
                        emb_mask=emb_mask,
                        emb_box=emb_box,
                        emb_text=emb_text,
                        meta=json.dumps(meta),
                    )

                    # debug images
                    debug_mask_path = (
                        embedded_items_dir
                        / f"{ref_label_slug}_frag_{frag_idx}_{ctrl_name}_mask.png"
                    )
                    debug_box_path = (
                        embedded_items_dir
                        / f"{ref_label_slug}_frag_{frag_idx}_{ctrl_name}_box.png"
                    )
                    img_mask.save(debug_mask_path)
                    img_box.save(debug_box_path)

                    # free per-ctrl stuff
                    del ctrl_img, img_mask, img_box, emb_mask, emb_box

                    print(
                        f"[x={x_id} | {ref_label_slug} | {ctrl_name}] frag {frag_idx}: "
                        f"saved embeddings -> {out_path.name}, "
                        f"debug imgs -> {debug_mask_path.name}, {debug_box_path.name}"
                    )

                # -------- fragment-level cleanup / CUDA cache flush --------
                frags_since_gc += 1
                if DEVICE.type == "cuda" and frags_since_gc >= FRAG_GC_INTERVAL:
                    torch.cuda.empty_cache()
                    gc.collect()
                    frags_since_gc = 0
                    print(
                        f"[INFO] Cleared CUDA cache after {FRAG_GC_INTERVAL} fragments."
                    )


if __name__ == "__main__":
    main()
