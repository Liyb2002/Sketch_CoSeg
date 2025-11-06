#!/usr/bin/env python
# Runs OWLv2 -> SAM with hard-coded inputs and saves masks + overlays.

import os, json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np
import cv2
import torch

from owl import detect_owlv2_boxes_counts  # local
from sam import SamRunner                  # local

# ---------- hard-coded IO ----------
IMAGE_PATH = "./1.png"
COMP_JSON  = "./components.json"
SAM_CKPT   = "./sam_vit_h_4b8939.pth"
OUT_DIR    = "./owl_sam_out"

# ---------- utils ----------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_items(json_path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      {"components":[{"count":2,"name":"Wheel"}, ...]}
      or legacy {"components":["Wheel","Seat",...]} -> converts to count=1 each.
    """
    payload = json.load(open(json_path))
    comps = payload.get("components", [])
    items: List[Dict[str, Any]] = []
    if comps and isinstance(comps[0], dict):
        for it in comps:
            name = str(it.get("name","")).strip()
            cnt = int(it.get("count", 1))
            if name and cnt >= 1:
                items.append({"name": name, "count": cnt})
    else:
        for name in comps:
            if isinstance(name, str) and name.strip():
                items.append({"name": name.strip(), "count": 1})
    if not items:
        raise ValueError("No valid components found in JSON.")
    return items

def slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

def color_for(lbl: str) -> tuple[int,int,int]:
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR

def save_mask_png(mask_bool: np.ndarray, out_path: str) -> None:
    cv2.imwrite(out_path, (mask_bool.astype(np.uint8) * 255))

def overlay_one(image_bgr: np.ndarray, mask_bool: np.ndarray, bgr_color: tuple[int,int,int]) -> np.ndarray:
    out = image_bgr.copy().astype(np.float32)
    col = np.array(bgr_color, np.float32)
    out[mask_bool] = 0.6 * out[mask_bool] + 0.4 * col
    return np.clip(out, 0, 255).astype(np.uint8)

def save_global_overlay(image_bgr: np.ndarray, masks: List[np.ndarray | None], labels: List[str], out_path: str) -> None:
    overlay = image_bgr.copy().astype(np.float32)
    for m, lbl in zip(masks, labels):
        if m is None: continue
        col = np.array(color_for(lbl), np.float32)
        overlay[m] = 0.6 * overlay[m] + 0.4 * col
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # legend
    y = 24
    for lbl in labels:
        col = color_for(lbl)
        cv2.rectangle(overlay, (10, y - 15), (30, y + 5), col, -1)
        cv2.putText(overlay, lbl, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        y += 22

    cv2.imwrite(out_path, overlay)

def main():
    ensure_dir(OUT_DIR)

    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_rgb.shape[:2]

    items = load_items(COMP_JSON)  # [{name,count}, ...]
    labels = [it["name"] for it in items]
    print(f"[INFO] Components: {items}")

    # ---- OWLv2 (counts + global non-overlap) ----
    results = detect_owlv2_boxes_counts(
        image_pil=image_pil,
        items=items,
        model_id="google/owlv2-large-patch14",
        use_tiles=True,
        tile_grid=3,
        tile_overlap=0.2,
        nms_iou=0.5,                    # NMS for candidate pooling
        score_thresholds=[0.30,0.25,0.20,0.15,0.10,0.07,0.05,0.03,0.01,0.0],
        enforce_no_overlap=True,        # global non-overlap constraint
    )
    # results: dict name -> {"boxes": (k,4), "scores": (k,), "thr": float}

    # ---- SAM: boxes -> masks (per component) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samr = SamRunner(sam_type="vit_h", sam_ckpt=SAM_CKPT, device=device)
    samr.set_image(image_rgb)

    masks_list: List[np.ndarray | None] = []
    for it in items:
        name = it["name"]
        entry = results.get(name, {"boxes": np.empty((0,4), np.float32), "scores": np.empty((0,), np.float32)})
        boxes = entry["boxes"]
        merged = None
        for b in boxes:
            m_best = samr.mask_from_box(b, (H, W))
            merged = m_best if merged is None else (merged | m_best)

        # save per-component mask + overlay
        if merged is None:
            masks_list.append(None)
        else:
            masks_list.append(merged)
            slug = slugify(name)
            mask_path = os.path.join(OUT_DIR, f"{slug}_mask.png")
            save_mask_png(merged, mask_path)

            per_overlay = overlay_one(image_bgr, merged, color_for(name))
            cv2.imwrite(os.path.join(OUT_DIR, f"{slug}_overlay.png"), per_overlay)

    # save global overlay
    save_global_overlay(image_bgr, masks_list, labels, os.path.join(OUT_DIR, "overlay.png"))
    json.dump(
        {"items": items, "results": {k: {"boxes": v["boxes"].tolist(), "scores": v["scores"].tolist()} for k,v in results.items()}},
        open(os.path.join(OUT_DIR, "results.json"), "w"),
        indent=2
    )
    print(f"[OK] wrote masks + overlays to {OUT_DIR}")

if __name__ == "__main__":
    main()
