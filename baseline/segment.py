#!/usr/bin/env python3
"""
segment_vlm_poly.py
VLM-only sketch segmentation:
  - Input image: sketch/0.png
  - Components:  sketch/out/0_components.json (or 0.components.json)
  - For each component, call OpenAI Vision to get a polygon (pixel coords).
  - Build a binary mask by filling the polygon and intersecting with a dilated edge map (from Canny).
  - Outputs:
      sketch/out/masks/<label>.png
      sketch/out/segmentation.json

Why this works better for sketches:
  - The VLM proposes geometry (polygon) instead of per-pixel logits.
  - We restrict the final mask to actual stroke pixels via edge ∩ polygon,
    avoiding the “solid blob” failure modes and random lines from CLIPSeg.
"""

import os, io, json, base64, time, re
from pathlib import Path
from typing import Dict, List, Tuple
import requests
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2

# --- Paths & config ---
IMG_PATH   = Path("sketch/0.png")
OUT_DIR    = Path("sketch/out")
COMP_JSON1 = OUT_DIR / "0_components.json"
COMP_JSON2 = OUT_DIR / "0.components.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = "gpt-4o"  # vision-capable & cost-effective

# Resize for VLM geometry extraction (keeps aspect; we map back to original size)
VLM_LONG_SIDE = 768

# Edge parameters (sketch-friendly)
CANNY_LOW, CANNY_HIGH = 40, 120  # tweak if needed
DILATE_KERNEL = 3                # widen lines slightly to form clean masks

# --- Helpers ---
def normalize_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\-\_/ ()]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s or "part"

def to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def load_components() -> List[str]:
    comp_json = COMP_JSON1 if COMP_JSON1.exists() else COMP_JSON2
    if not comp_json.exists():
        raise SystemExit(f"Missing components json: {COMP_JSON1} or {COMP_JSON2}")
    comps = json.loads(comp_json.read_text()).get("components", [])
    if not comps:
        raise SystemExit("Components list is empty in JSON")
    return comps

def pil_resize_keep_aspect(img: Image.Image, target_long: int) -> Tuple[Image.Image, float]:
    """Resize so that max(H,W)=target_long. Return resized image + scale factor (orig->resized)."""
    w, h = img.size
    scale = target_long / max(w, h)
    if scale >= 1.0:
        return img.copy(), 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC), scale

def call_openai_for_polygon(resized_img: Image.Image, label: str) -> Dict:
    """
    Ask VLM for a polygon around the given component label.
    Returns a dict:
      {
        "label": "...",
        "polygon": [[x0,y0], [x1,y1], ...]   # pixel coords in RESIZED image space
      }
    If polygon not possible, returns {"label":..., "bbox":[x,y,w,h]} in RESIZED space.
    """
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set")

    # Strong JSON-only extraction; no category guessing, only geometry for the named part
    system_prompt = (
        "You are a precise geometry extractor for line drawings. "
        "Given a target part name and an image, return a tight polygon enclosing that part. "
        "If a polygon is not possible, return a bounding box. "
        "Output must be a single JSON object with either:\n"
        '{ "label": <string>, "polygon": [[x,y],... (3 to 30 points)] }\n'
        "OR\n"
        '{ "label": <string>, "bbox": [x, y, w, h] }\n'
        "Coordinates are integer pixel indices in the provided image resolution. "
        "Do NOT include any other text."
    )
    user_prompt = (
        f"Target part name: '{label}'. "
        "Return ONLY JSON as specified. Use a polygon if you can; otherwise a bbox."
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type":"text","text": user_prompt},
                {"type":"image_url","image_url":{"url": to_data_uri(resized_img)}}
            ]}
        ]
    }
    r = requests.post(OPENAI_URL, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()

    # Parse JSON object
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("not a dict")
        return obj
    except Exception:
        # Best-effort extract JSON object
        try:
            start = text.find("{")
            end   = text.rfind("}")
            obj = json.loads(text[start:end+1])
            if not isinstance(obj, dict):
                raise ValueError("not a dict 2")
            return obj
        except Exception as e:
            raise RuntimeError(f"VLM did not return a valid JSON geometry for '{label}': {text[:200]}") from e

def polygon_to_mask(poly: List[List[float]], size_hw: Tuple[int,int]) -> np.ndarray:
    """Rasterize polygon to binary mask (H,W)."""
    h, w = size_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def bbox_to_mask(bbox: List[float], size_hw: Tuple[int,int]) -> np.ndarray:
    """Rasterize bbox [x,y,w,h] to binary mask (H,W)."""
    h, w = size_hw
    x, y, bw, bh = map(int, bbox)
    x2, y2 = max(0, x), max(0, y)
    x3, y3 = min(w, x2 + max(1, bw)), min(h, y2 + max(1, bh))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y2:y3, x2:x3] = 1
    return mask

def edges_from_sketch(orig_img: Image.Image) -> np.ndarray:
    """Canny edges + dilation -> (H,W) in {0,1}."""
    gray = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    # Normalize contrast a bit
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    if DILATE_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_KERNEL, DILATE_KERNEL))
        edges = cv2.dilate(edges, k, iterations=1)
    return (edges > 0).astype(np.uint8)

def save_mask(mask_np: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_np * 255).astype(np.uint8), mode="L").save(out_path)

# --- Main ---
def main():
    if not IMG_PATH.exists():
        raise SystemExit(f"Missing image: {IMG_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks_dir = OUT_DIR / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Read components
    components = load_components()
    components = components[:3]


    # Load image; build edge map in ORIGINAL size
    orig_img = Image.open(IMG_PATH).convert("RGB")
    orig_w, orig_h = orig_img.size
    edge_mask = edges_from_sketch(orig_img)  # (H,W) in {0,1}

    # For geometry extraction, use a resized version (better stability for VLM)
    resized_img, scale = pil_resize_keep_aspect(orig_img, VLM_LONG_SIDE)
    r_w, r_h = resized_img.size  # note PIL gives (W,H)

    label_to_file: Dict[str, str] = {}

    print(f"Segmenting {len(components)} components via VLM polygons ...")
    for label in tqdm(components, desc="Components", ncols=80):
        lbl = normalize_label(label)

        # 1) Ask VLM for geometry in RESIZED image space
        obj = call_openai_for_polygon(resized_img, lbl)

        # 2) Rasterize to RESIZED mask
        if "polygon" in obj and isinstance(obj["polygon"], list) and len(obj["polygon"]) >= 3:
            poly = obj["polygon"]
            mask_r = polygon_to_mask(poly, (r_h, r_w))
        elif "bbox" in obj and isinstance(obj["bbox"], list) and len(obj["bbox"]) == 4:
            mask_r = bbox_to_mask(obj["bbox"], (r_h, r_w))
        else:
            # If response malformed, skip this label
            # (You can also fallback to whole image or a heuristic)
            continue

        # 3) Upscale mask back to ORIGINAL size
        if scale != 1.0:
            mask_up = cv2.resize(mask_r.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_up = mask_r

        # 4) Intersect with edge map to keep only stroke pixels inside region
        final_mask = (mask_up > 0).astype(np.uint8) & edge_mask

        # 5) Small cleanups (optional)
        # remove tiny noise
        num_labels, labels_im = cv2.connectedComponents(final_mask)
        if num_labels > 1:
            # keep largest component
            areas = [(labels_im == i).sum() for i in range(1, num_labels)]
            if len(areas) > 0:
                keep_id = 1 + int(np.argmax(areas))
                final_mask = (labels_im == keep_id).astype(np.uint8)

        out_path = masks_dir / f"{lbl}.png"
        save_mask(final_mask, out_path)
        label_to_file[label] = str(out_path)

    # Write index
    (OUT_DIR / "segmentation.json").write_text(json.dumps({
        "image": str(IMG_PATH),
        "labels": components,
        "masks": label_to_file
    }, indent=2))

    print(f"\nDone. Wrote {len(label_to_file)} masks to {masks_dir}")
    print(f"Index: {OUT_DIR/'segmentation.json'}")

if __name__ == "__main__":
    main()
