#!/usr/bin/env python
# Full pipeline:
#   inputs/{x}.png + inputs/components_{x}.json
# -> outputs/{x}/realism_imgs/ctrl_*.png         (aligned to sketch)
# -> outputs/{x}/ctrl_*/{label}_mask.png        (SAM from OWL boxes)
# -> outputs/{x}/probabilistic_aggregate/*      (probability maps + mosaics)
# -> outputs/{x}/fragments/{label}/*            (per-fragment masks + overlays)
#
# Uses:
#   - make_realistic.generate_variants
#   - owl.detect_owlv2_boxes_counts
#   - sam.SamRunner
#   - probabilistic_combine.probabilistic_combine
#   - create_segments.create_segments

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image
import cv2
import torch

from owl import detect_owlv2_boxes_counts
from sam import SamRunner
from make_realistic import generate_variants
from probabilistic_combine import probabilistic_combine
from create_segments import create_segments
import helper

# ---------------- utils ----------------
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

def _load_items(components_json: str) -> List[Dict[str, Any]]:
    payload = json.load(open(components_json, "r"))
    comps = payload.get("components", [])
    items: List[Dict[str, Any]] = []
    if comps and isinstance(comps, list) and comps and comps and isinstance(comps[0], dict):
        for it in comps:
            n = str(it.get("name", "")).strip()
            c = int(it.get("count", 1))
            if n and c >= 1:
                items.append({"name": n, "count": c})
    else:
        for n in comps:
            if isinstance(n, str) and n.strip():
                items.append({"name": n.strip(), "count": 1})
    return items

def _get_object_label(components_json: str) -> str:
    """
    Prefer explicit object_name.
    Fallback: single-key dict (e.g. { "motobike": [...] }).
    Else: "object".
    """
    payload = json.load(open(components_json, "r"))
    if isinstance(payload, dict):
        if "object_name" in payload:
            return str(payload["object_name"]).strip()
        if "components" not in payload and len(payload) == 1:
            return list(payload.keys())[0]
    return "object"

def _color_for(lbl: str) -> tuple[int, int, int]:
    # BGR for cv2 overlays
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return int(c[2]), int(c[1]), int(c[0])

def _save_mask_png(mask_bool: np.ndarray, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, (mask_bool.astype(np.uint8) * 255))

def _sorted_input_pngs(inputs_dir: Path) -> List[Path]:
    pngs = list(inputs_dir.glob("*.png"))
    def key(p: Path):
        stem = p.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)
    return sorted(pngs, key=key)

# ---------------- main pipeline ----------------
def main():
    root = Path(__file__).resolve().parent
    inputs_dir = root / "inputs"
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    png_files = _sorted_input_pngs(inputs_dir)
    if not png_files:
        print(f"[WARN] No .png files found in {inputs_dir}")
        return

    for sketch_path in png_files:
        idx = sketch_path.stem
        components_json_path = inputs_dir / f"components_{idx}.json"

        if not components_json_path.exists():
            print(f"[WARN] Missing components_{idx}.json for {sketch_path.name}, skipping.")
            continue

        base_out_dir = outputs_dir / idx
        realism_img_dir = base_out_dir / "realism_imgs"
        base_out_dir.mkdir(parents=True, exist_ok=True)
        realism_img_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n==================== Processing {idx}.png ====================")

        # ---- load per-image config ----
        items = _load_items(str(components_json_path))
        if not items:
            print(f"[WARN] No components found in {components_json_path}, skipping.")
            continue

        object_label = _get_object_label(str(components_json_path))

        # ---- enriched OWL items: use object_name in query ----
        owl_items: List[Dict[str, Any]] = []
        for it in items:
            nm = it["name"]
            if object_label != "object":
                query = f"{nm} of a {object_label}"
            else:
                query = nm
            owl_items.append({
                "name": nm,
                "count": it["count"],
                "query": query,
            })

        # ---- style prompts for SDXL ----
        styles = [
            "high-resolution photograph of a {label} matching the sketch silhouette and pose exactly, "
            "studio product photo, realistic materials and textures, neutral soft lighting, in 21st century metallic style, "
            "subtle shadows, white background, sharp focus, no text, no watermark, no extra objects",

            "realistic photo of a {label}, matching the sketch silhouette and pose, "
            "natural colors, physically plausible lighting, simple real-world setting, in cyberpunk style, "
            "clear separation from white background, no stylization, no sketch lines, no extra objects, no text",

            "detailed realistic photo of a {label}, matching the sketch silhouette and pose, "
            "accurate proportions, real-world materials, studio lighting, green color dominant, white background, "
            "crisp edges, no fantasy elements, no text, no extra props",

            "high-detail real photo of a {label}, matching the sketch silhouette and pose exactly, "
            "captured with a full-frame DSLR camera, realistic textures and materials, "
            "visible metal wear, oil stains, authentic paint, subtle reflections, natural shadows, "
            "no extra objects, no text, no watermark",

            "clean Miyazaki-style illustration of a {label} from World War II, matching the sketch silhouette and pose, "
            "solid lineart, flat shading, simple natural colors, plain light background, "
            "no text, no sketch outline visible, object clearly readable and centered",
        ]
        style_prompts = [s.format(label=object_label) for s in styles]

        # ---- 1) generate aligned ctrl_* variants ----
        print("-------------------- generate_variant realistic images --------------------")
        ctrl_paths = generate_variants(
            input_path=str(sketch_path),
            out_dir=str(realism_img_dir),
            style_prompts=style_prompts,
            seed=2025,
        )

        # free VRAM (SDXL) best-effort before OWL+SAM
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # ---- 2) OWL detect + SAM masks per ctrl_i ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load sketch once
        sketch_rgb = np.array(Image.open(sketch_path).convert("RGB"))
        sketch_bgr_full = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR)

        sam_type = "vit_h"
        sam_ckpt = "./sam_vit_h_4b8939.pth"
        sam_runner = SamRunner(
            sam_type=sam_type,
            sam_ckpt=sam_ckpt,
            device=device,
            sketch_path=str(sketch_path),
        )

        for p in ctrl_paths:
            p = Path(p)
            sub = base_out_dir / p.stem          # e.g., outputs/x/ctrl_0
            sub.mkdir(parents=True, exist_ok=True)

            # --- OWL on this ctrl image ---
            img = Image.open(p).convert("RGB")
            results = detect_owlv2_boxes_counts(
                image_pil=img,
                items=owl_items,
                model_id="google/owlv2-large-patch14",
                use_tiles=True,
                tile_grid=3,
                tile_overlap=0.2,
                nms_iou=0.5,
                score_thresholds=(0.30,0.25,0.20,0.15,0.10,0.07,0.05,0.03,0.01,0.0),
                enforce_no_overlap=False,
            )

            # --- Save OWL detection results as JSON ---
            results_json_path = sub / "owl_results.json"
            try:
                serializable_results = helper._to_serializable(results)
                with open(results_json_path, "w") as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"Saved OWL results → {results_json_path}")
            except Exception as e:
                print(f"[WARN] Failed to save OWL results for {p.name}: {e}")


            det_count = sum(len(v["boxes"]) for v in results.values())
            print(f"{idx}/{p.stem}: {det_count} detections")

            # --- SAM on this ctrl image (box → merged mask per label) ---
            rgb = np.array(img)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            H, W = rgb.shape[:2]
            sam_runner.set_image(rgb)

            # resize sketch to ctrl size if needed
            if sketch_bgr_full.shape[:2] != (H, W):
                sketch_bgr = cv2.resize(sketch_bgr_full, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                sketch_bgr = sketch_bgr_full.copy()

            overlay_ctrl = bgr.astype(np.float32)
            overlay_skch = sketch_bgr.astype(np.float32)
            alpha = 0.4

            for it in items:
                label = it["name"]
                slug  = _slug(label)
                boxes = results.get(label, {}).get("boxes", np.empty((0, 4), np.float32))

                if boxes is None or len(boxes) == 0:
                    # still save an empty mask for consistency
                    empty = np.zeros((H, W), dtype=bool)
                    _save_mask_png(empty, str(sub / f"{slug}_mask.png"))
                    continue

                merged = np.zeros((H, W), dtype=bool)
                for bx in boxes:
                    mask = sam_runner.mask_from_box(np.asarray(bx, np.float32), (H, W))
                    merged |= mask

                _save_mask_png(merged, str(sub / f"{slug}_mask.png"))

                col = np.array(_color_for(label), np.float32)  # BGR
                overlay_ctrl[merged] = (1 - alpha) * overlay_ctrl[merged] + alpha * col
                overlay_skch[merged] = (1 - alpha) * overlay_skch[merged] + alpha * col

            overlay_ctrl = np.clip(overlay_ctrl, 0, 255).astype(np.uint8)
            overlay_skch = np.clip(overlay_skch, 0, 255).astype(np.uint8)

            cv2.imwrite(str(sub / "overlay_ctrl.png"), overlay_ctrl)
            cv2.imwrite(str(sub / "overlay_sketch.png"), overlay_skch)

            # per-ctrl cleanup
            try:
                del img, rgb, bgr, overlay_ctrl, overlay_skch
            except Exception:
                pass
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # ---- 3) probabilistic aggregation (per-label prob maps + overlays) ----
        probabilistic_combine(
            base_out_dir=str(base_out_dir),
            components_json=str(components_json_path),
            original_image_path=str(sketch_path),
        )

        # ---- 4) fragment extraction + visualization overlays ----
        create_segments(
            base_out_dir=str(base_out_dir),
            components_json=str(components_json_path),
            original_image_path=str(sketch_path),
        )

        # ---- per-sketch cleanup ----
        try:
            del sam_runner, sketch_rgb, sketch_bgr_full, ctrl_paths, items, owl_items
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("\n[DONE] Processed all valid inputs in 'inputs/'.")

if __name__ == "__main__":
    main()
