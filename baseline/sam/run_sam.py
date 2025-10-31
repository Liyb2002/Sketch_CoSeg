#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 2.1 — CPU ONLY — low-res sketch segmentation
No Hydra overrides; CUDA fully disabled; safe on macOS CPU builds of PyTorch.
"""

import os, sys, json, time
from pathlib import Path
import numpy as np
import cv2

# ---------- Force pure CPU BEFORE importing torch/sam2 ----------
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # hide GPUs
os.environ["SAM2_DEVICE"] = "cpu"           # hint some builds respect
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch

# Hard-stop any CUDA path (even if something tries to call it)
torch.cuda.is_available   = lambda: False
torch.cuda.device_count   = lambda: 0
torch.cuda.current_device = lambda: -1
torch.cuda.get_device_name= lambda *_: "CPU"
torch.cuda.empty_cache    = lambda: None
torch.cuda._initialized   = True  # fool internal CUDA checks

# Force all checkpoint loads to CPU
_orig_torch_load = torch.load
def _cpu_load(*args, **kwargs):
    kwargs["map_location"] = "cpu"
    return _orig_torch_load(*args, **kwargs)
torch.load = _cpu_load  # type: ignore

# Route any accidental .to("cuda") to CPU
_orig_module_to = torch.nn.Module.to
def _safe_to(self, *args, **kwargs):
    if len(args) >= 1 and isinstance(args[0], str) and args[0].lower().startswith("cuda"):
        args = ("cpu",) + args[1:]
    dev = kwargs.get("device", None)
    if isinstance(dev, str) and dev.lower().startswith("cuda"):
        kwargs["device"] = "cpu"
    return _orig_module_to(self, *args, **kwargs)
torch.nn.Module.to = _safe_to  # type: ignore

# ---------- Hydra / SAM2 (no overrides) ----------
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ---------- Paths / config ----------
INPUT_DIR  = Path.cwd()               # .png images here
OUTPUT_DIR = Path.cwd() / "sam_out"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CFG_DIR = Path("/Users/yuanboli/Documents/GitHub/Sketch_CoSeg/baseline/sam/sam2_ckpts").resolve()
CFG_NAME = "sam2.1_hiera_l.yaml"     # local file in CFG_DIR
SAM2_CHECKPOINT = str(CFG_DIR / "sam2.1_hiera_large.pt")

# CPU-friendly MaskGen + low resolution
MASKGEN_KW = dict(
    points_per_side=16,
    pred_iou_thresh=0.65,
    stability_score_thresh=0.65,
    box_nms_thresh=0.8,
    min_mask_region_area=16,
    output_mode="binary_mask",
    use_m2m=False,
    multimask_output=False,
    crop_n_layers=0,
)
LONG_SIDE_MAX = 512  # speed

# ---------- Helpers ----------
def _resize_for_speed(img_bgr, long_max=LONG_SIDE_MAX):
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= long_max:
        return img_bgr, 1.0
    scale = long_max / float(long_side)
    new_w, new_h = int(round(w*scale)), int(round(h*scale))
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

def _masks_to_label_png(masks, h, w):
    label = np.zeros((h, w), np.uint16)
    if not masks:
        return label
    for idx, m in enumerate(sorted(
        masks, key=lambda m: (float(m.get("predicted_iou", 0.0)), int(m["area"])), reverse=True
    ), 1):
        seg = m["segmentation"]
        if isinstance(seg, np.ndarray):
            label[seg.astype(bool)] = idx
    return label

def _draw_contours_overlay(img_bgr, masks):
    vis = img_bgr.copy()
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        cs, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 255, 0), 1)
    return vis

def _rle_encode_boolmask(mask_bool):
    try:
        from pycocotools import mask as m
        rle = m.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        arr = np.asfortranarray(mask_bool).reshape((-1,), order="F").astype(np.uint8)
        counts, prev, run = [], 0, 0
        for v in arr:
            if v != prev:
                counts.append(run)
                run, prev = 1, v
            else:
                run += 1
        counts.append(run)
        return {"size": list(mask_bool.shape), "counts": counts}

def _assert_requirements():
    cfg_path = (CFG_DIR / CFG_NAME)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")
    ckpt = Path(SAM2_CHECKPOINT)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if "sam2.1_" not in ckpt.name:
        raise RuntimeError(f"Checkpoint looks non-2.1: {ckpt.name}")

# ---------- Main ----------
def main():
    _assert_requirements()

    print(f"[INFO] CPU-only mode")
    print(f"[INFO] Config:     {CFG_DIR / CFG_NAME}")
    print(f"[INFO] Checkpoint: {SAM2_CHECKPOINT}")
    print(f"[INFO] Long side:  {LONG_SIDE_MAX}px")

    # Compose config from local dir (no package subpaths, no overrides)
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CFG_DIR)):
        sam2_model = build_sam2(CFG_NAME, SAM2_CHECKPOINT)
        sam2_model.to("cpu")  # final safeguard

    mask_generator = SAM2AutomaticMaskGenerator(sam2_model, **MASKGEN_KW)

    images = sorted([p for p in INPUT_DIR.glob("*.png")],
                    key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not images:
        print(f"[INFO] No .png images in {INPUT_DIR}")
        return

    for img_path in images:
        print(f"\n[IMG] {img_path.name}")
        t0 = time.time()
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("   ⚠️  unreadable — skipped")
            continue

        # sketch-friendly prep
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)
        thr = cv2.dilate(thr, np.ones((2, 2), np.uint8), 1)
        img_bgr = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

        # downscale for CPU speed
        h0, w0 = img_bgr.shape[:2]
        small_bgr, scale = _resize_for_speed(img_bgr, LONG_SIDE_MAX)
        small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

        masks_small = mask_generator.generate(small_rgb)
        print(f"   masks: {len(masks_small)}  (scale={scale:.2f})")

        # upscale masks back
        masks = []
        for m in masks_small:
            seg_small = m["segmentation"].astype(np.uint8)
            seg_up = cv2.resize(seg_small, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(bool)
            m2 = m.copy()
            m2["segmentation"] = seg_up
            m2["area"] = int(seg_up.sum())
            masks.append(m2)

        # outputs
        label = _masks_to_label_png(masks, h0, w0)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_mask.png"), label.astype(np.uint16))
        overlay = _draw_contours_overlay(img_bgr, masks)
        cv2.imwrite(str(OUTPUT_DIR / f"{img_path.stem}_overlay.png"), overlay)

        out = []
        for i, m in enumerate(masks, 1):
            x, y, bw, bh = m["bbox"]
            out.append({
                "id": i,
                "rle": _rle_encode_boolmask(m["segmentation"]),
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(m["area"]),
                "predicted_iou": float(m.get("predicted_iou", 0.0)),
                "stability_score": float(m.get("stability_score", 0.0)),
            })
        with open(OUTPUT_DIR / f"{img_path.stem}_masks.json", "w") as f:
            json.dump({"image": img_path.name, "height": h0, "width": w0, "masks": out}, f, indent=2)
        print(f"   ✅ done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
