#!/usr/bin/env python
# Runs OWLv2 -> SAM on a sketch with hard-coded inputs.

import os, json
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch

from owl import detect_owlv2_boxes   # local
from sam import SamRunner            # local

# ---------- hard-coded IO ----------
IMAGE_PATH = "./1.png"
COMP_JSON  = "./components.json"
SAM_CKPT   = "./sam_vit_h_4b8939.pth"
OUT_DIR    = "./owl_sam_out"

# ---------- small utils ----------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_component_names(json_path: str) -> list[str]:
    payload = json.load(open(json_path))
    comps = payload.get("components", [])
    names = []
    for it in comps:
        if isinstance(it, str):
            n = it.strip()
            if n: names.append(n)
        elif isinstance(it, dict):
            n = str(it.get("name", "")).strip()
            if n: names.append(n)
    # keep order, dedupe
    dedup = []
    for n in names:
        if n not in dedup:
            dedup.append(n)
    return dedup

def save_mask_png(mask_bool: np.ndarray, out_path: str) -> None:
    cv2.imwrite(out_path, (mask_bool.astype(np.uint8) * 255))

def color_for(lbl: str) -> tuple[int,int,int]:
    # deterministic BGR
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))

def save_overlay(image_bgr: np.ndarray, masks: list[np.ndarray | None], labels: list[str], out_path: str) -> None:
    overlay = image_bgr.copy().astype(np.float32)
    for m, lbl in zip(masks, labels):
        if m is None: continue
        col = np.array(color_for(lbl), np.float32)
        overlay[m] = 0.6 * overlay[m] + 0.4 * col
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # quick legend
    y = 24
    for lbl in labels:
        col = color_for(lbl)
        cv2.rectangle(overlay, (10, y - 15), (30, y + 5), col, -1)
        cv2.putText(overlay, lbl, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        y += 22

    cv2.imwrite(out_path, overlay)

def main():
    ensure_dir(OUT_DIR)

    # Load image (RGB for models, BGR for OpenCV)
    image_pil = Image.open(IMAGE_PATH).convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_rgb.shape[:2]

    # Load labels
    labels = load_component_names(COMP_JSON)
    print(f"[INFO] Components: {labels}")

    # ---- OWLv2: get boxes per label (variable count, best-effort) ----
    # You can tweak model_id / tiling defaults here if you want.
    owl_results = detect_owlv2_boxes(
        image_pil=image_pil,
        labels=labels,
        model_id="google/owlv2-large-patch14",
        use_tiles=True,
        tile_grid=3,
        tile_overlap=0.2,
        nms_iou=0.5,
        score_thresholds=[0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01, 0.0],
        max_per_label=5,
    )
    # owl_results: dict {label: {"boxes": (K,4) float32, "scores": (K,), "thr": float}}

    # ---- SAM: convert boxes -> masks ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samr = SamRunner(sam_type="vit_h", sam_ckpt=SAM_CKPT, device=device)
    samr.set_image(image_rgb)

    masks_per_label: list[np.ndarray | None] = []
    manifest = {"items": []}

    for lbl in labels:
        entry = owl_results.get(lbl, {"boxes": np.empty((0,4), np.float32), "scores": np.empty((0,), np.float32), "thr": None})
        boxes = entry["boxes"]
        scores = entry["scores"]

        merged = None
        for b in boxes:
            m_best = samr.mask_from_box(b, (H, W))  # bool HxW
            if merged is None:
                merged = m_best.copy()
            else:
                merged |= m_best

        if merged is None:
            masks_per_label.append(None)
            manifest["items"].append({
                "name": lbl, "boxes_xyxy": [], "scores": [], "mask_path": None
            })
        else:
            masks_per_label.append(merged)
            slug = "".join(c.lower() if c.isalnum() else "_" for c in lbl).strip("_")
            mask_path = os.path.join(OUT_DIR, f"{slug}_mask.png")
            save_mask_png(merged, mask_path)
            manifest["items"].append({
                "name": lbl,
                "boxes_xyxy": boxes.tolist(),
                "scores": scores.tolist(),
                "mask_path": mask_path
            })

    # Save overlay
    overlay_path = os.path.join(OUT_DIR, "overlay.png")
    save_overlay(image_bgr, masks_per_label, labels, overlay_path)
    json.dump(manifest, open(os.path.join(OUT_DIR, "results.json"), "w"), indent=2)

    print(f"[OK] Wrote masks + overlay to {OUT_DIR}")
    print(f"      overlay: {overlay_path}")

if __name__ == "__main__":
    main()
