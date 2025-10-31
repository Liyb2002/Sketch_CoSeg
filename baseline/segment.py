#!/usr/bin/env python3
"""
segment_gpt5.py — sketch segmentation using GPT-5 vision

Pipeline (first 3 components):
  1) Ask GPT-5 for a polygon or bbox per component.
  2) Rasterize to a mask (resized space) → scale back to original.
  3) Intersect with Canny edges to hug stroke pixels.

Outputs:
  sketch/out/masks/<label>.png
  sketch/out/segmentation.json
"""

import os, io, json, base64, time, re
from pathlib import Path
from typing import Dict, List, Tuple
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# ---------- paths & config ----------
IMG_PATH   = Path("sketch/0.png")
OUT_DIR    = Path("sketch/out")
COMP_JSON1 = OUT_DIR / "0_components.json"
COMP_JSON2 = OUT_DIR / "0.components.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_VLM_MODEL", "gpt-5")  # or "gpt-5-mini"
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"

# resize long side for VLM geometry extraction
VLM_LONG_SIDE = 768
# edge extraction
CANNY_LOW, CANNY_HIGH = 40, 120
DILATE_KERNEL = 3
# only segment first N components
MAX_COMPONENTS = 3


# ---------- helpers ----------
def normalize_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\\-_/ ()]+", "", s)
    s = re.sub(r"\\s+", " ", s)
    return s or "part"


def to_data_uri_jpeg(img: Image.Image, *, max_side=768, quality=75) -> str:
    """Resize to <=max_side and encode as JPEG data URI."""
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def load_components() -> List[str]:
    comp_json = COMP_JSON1 if COMP_JSON1.exists() else COMP_JSON2
    if not comp_json.exists():
        raise SystemExit(f"Missing components json: {COMP_JSON1} or {COMP_JSON2}")
    comps = json.loads(comp_json.read_text()).get("components", [])
    if not comps:
        raise SystemExit("Components list is empty in JSON")
    return comps


def pil_resize_keep_aspect(img: Image.Image, target_long: int) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = target_long / max(w, h)
    if scale >= 1.0:
        return img.copy(), 1.0
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.BICUBIC), scale


def polygon_to_mask(poly: List[List[float]], size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def bbox_to_mask(bbox: List[float], size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    x, y, bw, bh = map(int, bbox)
    x2, y2 = max(0, x), max(0, y)
    x3, y3 = min(w, x2 + max(1, bw)), min(h, y2 + max(1, bh))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y2:y3, x2:x3] = 1
    return mask


def edges_from_sketch(orig_img: Image.Image) -> np.ndarray:
    """Return (H,W) binary edge map from Canny + dilation."""
    gray = cv2.cvtColor(np.array(orig_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    if DILATE_KERNEL > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_KERNEL, DILATE_KERNEL))
        edges = cv2.dilate(edges, k, iterations=1)
    return (edges > 0).astype(np.uint8)


def save_mask(mask_np: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask_np * 255).astype(np.uint8), mode="L").save(out_path)


# ---------- GPT-5 call ----------
def call_gpt5_for_polygon(resized_img: Image.Image, label: str) -> dict:
    """
    Ask GPT-5 for polygon/bbox geometry for a given part label.
    Returns dict with {"label":..., "polygon":[[x,y],...]} or {"label":..., "bbox":[x,y,w,h]}.
    """
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY not set")

    data_uri = to_data_uri_jpeg(resized_img, max_side=max(resized_img.size), quality=75)

    system_prompt = (
        "You extract geometry for a named part in a line drawing. "
        "Return ONLY JSON with either:\n"
        '{\"label\": <string>, \"polygon\": [[x,y], ...] }\n'
        "OR\n"
        '{\"label\": <string>, \"bbox\": [x, y, w, h] }\n'
        "Coordinates are integers in the provided image resolution. No text besides the JSON."
    )
    user_prompt = (
        f"Target part name: '{label}'. "
        "Return a tight polygon (3–30 points) or a tight bounding box if a polygon is not possible."
    )

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ],
        "response_format": {"type": "text"}
    }

    for attempt in range(3):
        r = requests.post(OPENAI_URL, headers=headers, data=json.dumps(payload), timeout=90)
        if r.status_code == 200:
            break
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (attempt + 1))
            continue
        print("OpenAI error:", r.status_code, r.text[:300])
        r.raise_for_status()

    out = r.json()
    text = out["choices"][0]["message"]["content"].strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    raise RuntimeError(f"GPT-5 did not return valid JSON for '{label}': {text[:200]}")


# ---------- main ----------
def main():
    if not IMG_PATH.exists():
        raise SystemExit(f"Missing image: {IMG_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    masks_dir = OUT_DIR / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    components = load_components()[16:19]

    orig_img = Image.open(IMG_PATH).convert("RGB")
    orig_w, orig_h = orig_img.size
    edge_mask = edges_from_sketch(orig_img)

    resized_img, scale = pil_resize_keep_aspect(orig_img, VLM_LONG_SIDE)
    r_w, r_h = resized_img.size

    label_to_file: Dict[str, str] = {}

    print(f"Segmenting {len(components)} components via GPT-5 ({OPENAI_MODEL}) ...")
    for label in tqdm(components, desc="Components", ncols=80):
        lbl = normalize_label(label)
        try:
            obj = call_gpt5_for_polygon(resized_img, lbl)
        except Exception as e:
            print(f"[warn] failed for {lbl}: {e}")
            continue

        if "polygon" in obj and isinstance(obj["polygon"], list) and len(obj["polygon"]) >= 3:
            mask_r = polygon_to_mask(obj["polygon"], (r_h, r_w))
        elif "bbox" in obj and isinstance(obj["bbox"], list) and len(obj["bbox"]) == 4:
            mask_r = bbox_to_mask(obj["bbox"], (r_h, r_w))
        else:
            continue

        # scale back to original
        if scale != 1.0:
            mask_up = cv2.resize(mask_r, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_up = mask_r

        final_mask = (mask_up > 0).astype(np.uint8) & edge_mask

        # keep largest region
        num_labels, labels_im = cv2.connectedComponents(final_mask)
        if num_labels > 1:
            areas = [(labels_im == i).sum() for i in range(1, num_labels)]
            if areas:
                keep_id = 1 + int(np.argmax(areas))
                final_mask = (labels_im == keep_id).astype(np.uint8)

        out_path = masks_dir / f"{lbl}.png"
        save_mask(final_mask, out_path)
        label_to_file[label] = str(out_path)

    (OUT_DIR / "segmentation.json").write_text(json.dumps({
        "image": str(IMG_PATH),
        "labels": components,
        "masks": label_to_file
    }, indent=2))

    print(f"\nDone. Wrote {len(label_to_file)} masks to {masks_dir}")
    print(f"Index: {OUT_DIR / 'segmentation.json'}")


if __name__ == "__main__":
    main()
