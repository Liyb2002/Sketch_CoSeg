#!/usr/bin/env python
# Orchestrator for: realism generation -> OWL-V2 detection -> SAM masks per variant -> union.
# Prints only the detection count per ctrl_x.png.

import os, re, json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import cv2
import torch

# your baseline detector (DO NOT MODIFY)
from owl import detect_owlv2_boxes_counts

# your SAM wrapper (we'll use SamRunner directly)
from sam import SamRunner

# new helpers we added earlier
from make_realistic import generate_variants
from combine_masks import combine_all

# ---------------- utils ----------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def _load_items(components_json: str) -> List[Dict[str, Any]]:
    # *** unchanged behavior ***
    payload = json.load(open(components_json, "r"))
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

def _get_object_label(components_json: str) -> str:
    """
    Get global object label for prompts without breaking anything.

    Works with:
      { "motobike": [ ... ] }   -> returns "motobike"
      { "components": [...] }   -> falls back to "object" unless you later add 'object_name'
    """
    payload = json.load(open(components_json, "r"))
    if isinstance(payload, dict):
        if "components" not in payload and len(payload) == 1:
            # your motobike-style JSON
            return list(payload.keys())[0]
        if "object_name" in payload:
            return str(payload["object_name"])
    return "object"

def _color_for(lbl: str) -> tuple[int,int,int]:
    # BGR for cv2 overlays
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return int(c[2]), int(c[1]), int(c[0])

def _save_mask_png(mask_bool: np.ndarray, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, (mask_bool.astype(np.uint8) * 255))

def _overlay_faded(base_bgr: np.ndarray, mask_bool: np.ndarray, color_bgr: tuple[int,int,int]) -> np.ndarray:
    """50% faded background + 40% color on masked pixels."""
    out = (base_bgr.astype(np.float32) * 0.5)
    col = np.array(color_bgr, np.float32)
    out[mask_bool] = 0.6 * out[mask_bool] + 0.4 * col
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------------- main pipeline ----------------
def main():
    # ---- paths (only here) ----
    sketch_path       = "0.png"
    components_json   = "components.json"
    base_out_dir      = "owl_sam_output"
    realism_img_dir   = os.path.join(base_out_dir, "realism_imgs")
    Path(base_out_dir).mkdir(parents=True, exist_ok=True)
    Path(realism_img_dir).mkdir(parents=True, exist_ok=True)

    # ---- load items (same as before) ----
    items = _load_items(components_json)

    # ---- get object label for prompts (new) ----
    object_label = _get_object_label(components_json)
    # e.g. from { "motobike": [...] } => "motobike"

    # ---- multi-style prompts (now labeled) ----
    styles = [
        "high-resolution photograph of a {label} matching the sketch silhouette and pose exactly, "
        "studio product photo, realistic materials and textures, neutral soft lighting, in 21st century metalatic style"
        "subtle shadows, white background, sharp focus, no text, no watermark, no extra objects",

        "realistic photo of a {label}, matching the sketch silhouette and pose, "
        "natural colors, physically plausible lighting, simple real-world setting, in cyberpunk style"
        "clear separation from white background, no stylization, no sketch lines, no extra objects, no text",

        "detailed realistic photo of a {label}, matching the sketch silhouette and pose, "
        "accurate proportions, real-world materials, studio lighting, grean color domainate, white background, "
        "crisp edges, no fantasy elements, no text, no extra props",

        "high-detail real photo of a {label}, matching the sketch silhouette and pose exactly, "
        "captured with a full-frame DSLR camera, realistic textures and materials,"
        "visible metal wear, oil stains, authentic paint, subtle reflections, natural shadows", 

        "clean Miyazaki-style illustration of a {label} from world war II, matching the sketch silhouette and pose, "
        "solid lineart, flat shading, simple natural colors, plain light background, "
        "no text, no sketch outline visible, object clearly readable and centered",
    ]

    style_prompts = [s.format(label=object_label) for s in styles]

    # ---- 1) generate ctrl_* variants ----
    print("--------------------generate_variant realistic images--------------------")
    ctrl_paths = generate_variants(
        input_path=sketch_path,
        out_dir=realism_img_dir,
        style_prompts=style_prompts,
        seed=2025,
    )

    # free VRAM between steps
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            torch.cuda.empty_cache()
    except Exception:
        pass

    # ---- 2) OWL detect + SAM masks per ctrl_i ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load sketch once (RGB/BGR) and keep for overlays
    sketch_rgb = np.array(Image.open(sketch_path).convert("RGB"))
    sketch_bgr_full = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR)

    # prepare SAM once (you can change type/ckpt here)
    sam_type = "vit_h"
    sam_ckpt = "./sam_vit_h_4b8939.pth"
    sam_runner = SamRunner(sam_type=sam_type, sam_ckpt=sam_ckpt, device=device)
    # let SamRunner know the sketch for overlay_sketch.png writes later
    sam_runner.set_sketch(sketch_rgb)

    for p in ctrl_paths:
        sub = Path(base_out_dir) / Path(p).stem              # owl_sam_output/ctrl_0
        sub.mkdir(parents=True, exist_ok=True)

        # --- OWL on this ctrl image ---
        img = Image.open(p).convert("RGB")
        results = detect_owlv2_boxes_counts(
            image_pil=img,
            items=items,
            model_id="google/owlv2-large-patch14",
            use_tiles=True,
            tile_grid=3,
            tile_overlap=0.2,
            nms_iou=0.5,
            score_thresholds=(0.30,0.25,0.20,0.15,0.10,0.07,0.05,0.03,0.01,0.0),
            enforce_no_overlap=False,
        )

        # ---- print detection count for this ctrl_x.png ----
        det_count = sum(len(v["boxes"]) for v in results.values())
        print(f"{Path(p).stem}: {det_count} detections")

        # --- SAM on this ctrl image (box → mask) ---
        rgb = np.array(img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = rgb.shape[:2]
        sam_runner.set_image(rgb)

        # prepare a sketch base (resize to ctrl size if needed) for per-variant overlay
        if sketch_bgr_full.shape[:2] != (H, W):
            sketch_bgr = cv2.resize(sketch_bgr_full, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            sketch_bgr = sketch_bgr_full.copy()

        # ---------- overlay logic: keep originals, tint only masks ----------
        overlay_ctrl = bgr.copy().astype(np.float32)
        overlay_skch = sketch_bgr.copy().astype(np.float32)
        alpha = 0.4  # tint strength on masked pixels only

        for it in items:
            label = it["name"]
            slug  = _slug(label)
            boxes = results.get(label, {}).get("boxes", np.empty((0,4), np.float32))

            if boxes is None or len(boxes) == 0:
                # still save an empty mask for consistency
                empty = np.zeros((H, W), dtype=bool)
                _save_mask_png(empty, str(sub / f"{slug}_mask.png"))
                continue

            # merge multiple boxes for the same label
            merged = np.zeros((H, W), dtype=bool)
            for bx in boxes:
                mask = sam_runner.mask_from_box(np.asarray(bx, np.float32), (H, W))
                merged |= mask

            # save merged mask
            _save_mask_png(merged, str(sub / f"{slug}_mask.png"))

            # tint only the masked pixels with a label-specific color
            col = np.array(_color_for(label), np.float32)  # BGR
            overlay_ctrl[merged] = (1 - alpha) * overlay_ctrl[merged] + alpha * col
            overlay_skch[merged] = (1 - alpha) * overlay_skch[merged] + alpha * col
        # ---------- END overlay ----------

        overlay_ctrl = np.clip(overlay_ctrl, 0, 255).astype(np.uint8)
        overlay_skch = np.clip(overlay_skch, 0, 255).astype(np.uint8)

        # ✅ overlays are saved here
        cv2.imwrite(str(sub / "overlay_ctrl.png"), overlay_ctrl)
        cv2.imwrite(str(sub / "overlay_sketch.png"), overlay_skch)

    # ---- 3) union across ctrl_* into final_output on original sketch ----
    combine_all(
        base_out_dir=base_out_dir,
        components_json=components_json,
        original_image_path=sketch_path,
    )

if __name__ == "__main__":
    main()
