#!/usr/bin/env python
# Hard-coded pipeline:
#   0) make_realistic: 0.png -> ctrl.png (photoreal)
#   1) OWLv2+SAM on ctrl.png, APPLY masks to original 0.png  -> owl_sam_out/realism/
#   2) OWLv2+SAM on 0.png,   APPLY masks to original 0.png  -> owl_sam_out/sketch/
# Only prints two summary lines at the very end.

import io, os, sys, json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import cv2
import torch

from owl import detect_owlv2_boxes_counts
from sam import SamRunner
import make_realistic

# ----- hard-coded IO -----
IMAGE_PATH   = "./0.png"
REAL_PATH    = "./ctrl.png"           # produced by make_realistic
COMP_JSON    = "./components.json"
SAM_CKPT     = "./sam_vit_h_4b8939.pth"

OUT_ROOT     = "./owl_sam_out"
OUT_REALISM  = "./owl_sam_out/realism"
OUT_SKETCH   = "./owl_sam_out/sketch"

RUN1_PREFIX  = "realism"              # labels for filenames
RUN2_PREFIX  = "sketch"

MODEL_ID     = "google/owlv2-large-patch14"
USE_TILES    = True
TILE_GRID    = 3
TILE_OVERLAP = 0.2
NMS_IOU      = 0.5
THR_SWEEP    = [0.30,0.25,0.20,0.15,0.10,0.07,0.05,0.03,0.01,0.0]

# ----- tiny utils -----
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def load_items(json_path: str) -> List[Dict[str, Any]]:
    payload = json.load(open(json_path))
    comps = payload.get("components", [])
    items: List[Dict[str, Any]] = []
    if comps and isinstance(comps[0], dict):
        for it in comps:
            n = str(it.get("name","")).strip()
            c = int(it.get("count", 1))
            if n and c >= 1: items.append({"name": n, "count": c})
    else:
        for n in comps:
            if isinstance(n, str) and n.strip():
                items.append({"name": n.strip(), "count": 1})
    return items

def slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

def color_for(lbl: str) -> tuple[int,int,int]:
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR

def save_mask_png(mask_bool: np.ndarray, out_path: str) -> None:
    cv2.imwrite(out_path, (mask_bool.astype(np.uint8) * 255))

def overlay_mask(image_bgr: np.ndarray, mask_bool: np.ndarray, bgr_color: tuple[int,int,int]) -> np.ndarray:
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
    y = 24
    for lbl in labels:
        col = color_for(lbl)
        cv2.rectangle(overlay, (10, y - 15), (30, y + 5), col, -1)
        cv2.putText(overlay, lbl, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
        y += 22
    cv2.imwrite(out_path, overlay)

class _Silence:
    def __enter__(self):
        self._null = io.StringIO()
        self._stdout_cm = redirect_stdout(self._null)
        self._stderr_cm = redirect_stderr(self._null)
        self._stdout_cm.__enter__()
        self._stderr_cm.__enter__()
        return self
    def __exit__(self, exc_type, exc, tb):
        self._stderr_cm.__exit__(exc_type, exc, tb)
        self._stdout_cm.__exit__(exc_type, exc, tb)

# ----- detection + masks -----
def _owl_sam_once_for_image(
    det_pil: Image.Image,
    apply_bgr: np.ndarray,
    items: List[Dict[str,Any]],
    out_dir: str,
    prefix: str,
    sam_ckpt: str
) -> int:
    """
    det_pil: PIL RGB used for OWL detection
    apply_bgr: BGR image to APPLY masks to (the original sketch)
    Returns: number of components with >=1 detected box
    """
    H, W = apply_bgr.shape[:2]
    labels = [it["name"] for it in items]

    with _Silence():
        results = detect_owlv2_boxes_counts(
            image_pil=det_pil,
            items=items,
            model_id=MODEL_ID,
            use_tiles=USE_TILES,
            tile_grid=TILE_GRID,
            tile_overlap=TILE_OVERLAP,
            nms_iou=NMS_IOU,
            score_thresholds=THR_SWEEP,
            enforce_no_overlap=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samr = SamRunner(sam_type="vit_h", sam_ckpt=sam_ckpt, device=device)
    # SAM should see the same image we will overlay onto (the sketch), so masks align 1:1
    samr.set_image(cv2.cvtColor(apply_bgr, cv2.COLOR_BGR2RGB))

    masks_list = []
    for it in items:
        nm = it["name"]
        entry = results.get(nm, {"boxes": np.empty((0,4), np.float32)})
        boxes = entry["boxes"]
        merged = None
        for b in boxes:
            m = samr.mask_from_box(b, (H, W))
            merged = m if merged is None else (merged | m)
        masks_list.append(merged if merged is not None else None)

    # per-component artifacts
    for it, m in zip(items, masks_list):
        nm = it["name"]; slug = slugify(nm)
        if m is None: continue
        save_mask_png(m, os.path.join(out_dir, f"{prefix}_{slug}_mask.png"))
        ov = overlay_mask(apply_bgr, m, color_for(nm))
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{slug}_overlay.png"), ov)

    # global overlay
    save_global_overlay(apply_bgr, masks_list, labels, os.path.join(out_dir, f"{prefix}_overlay.png"))

    # count how many components got >= 1 box
    detected = sum(1 for it in items if len(results.get(it["name"], {}).get("boxes", [])) >= 1)
    return detected

def main():
    # folders
    ensure_dir(OUT_ROOT)
    ensure_dir(OUT_REALISM)
    ensure_dir(OUT_SKETCH)

    # 0) make photoreal ctrl.png (quiet)
    with _Silence():
        make_realistic.run()

    # load images
    sketch_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    sketch_bgr = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR)
    ctrl_pil   = Image.open(REAL_PATH).convert("RGB")   # same size as sketch
    sketch_pil = Image.open(IMAGE_PATH).convert("RGB")

    items = load_items(COMP_JSON)
    total = len(items)

    # 1) OWL+SAM on ctrl.png (apply masks to sketch) -> owl_sam_out/realism
    det_real = _owl_sam_once_for_image(ctrl_pil, sketch_bgr, items, OUT_REALISM, RUN1_PREFIX, SAM_CKPT)

    # 2) OWL+SAM on 0.png (apply masks to sketch) -> owl_sam_out/sketch
    det_skch = _owl_sam_once_for_image(sketch_pil, sketch_bgr, items, OUT_SKETCH, RUN2_PREFIX, SAM_CKPT)

    # Final, minimal prints:
    print(f"OWL-Det (realism): {det_real} / {total}")
    print(f"OWL-Det (sketch): {det_skch} / {total}")

if __name__ == "__main__":
    main()
