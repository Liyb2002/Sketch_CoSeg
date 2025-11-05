#!/usr/bin/env python
import argparse, json, os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from PIL import Image
import cv2
import torchvision
from transformers import AutoProcessor, GroundingDinoForObjectDetection

# ---------------- utils ----------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")

def color_for(lbl: str) -> tuple[int,int,int]:
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR

def load_components(json_path: str) -> List[str]:
    with open(json_path, "r") as f:
        payload = json.load(f)
    comps = payload.get("components", [])
    return [c.strip() for c in comps if isinstance(c, str) and c.strip()]

def draw_boxes(img_bgr: np.ndarray, boxes: np.ndarray, label: str, scores=None, color=None, th=2):
    vis = img_bgr.copy()
    color = color or color_for(label)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, th)
        txt = label if scores is None else f"{label}: {scores[i]:.2f}"
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytxt = max(0, y1 - 6)
        cv2.rectangle(vis, (x1, ytxt - tht - 4), (x1 + tw + 6, ytxt), (0,0,0), -1)
        cv2.putText(vis, txt, (x1+3, ytxt-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
    return vis

# -------------- sketch helpers --------------
def to_gray_01(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g

def stroke_density(gray01: np.ndarray, box_xyxy: np.ndarray, thresh: float = 0.8) -> float:
    """Estimate % of 'ink' pixels inside box (ink = dark). thresh is on [0..1] gray."""
    x1,y1,x2,y2 = [int(round(v)) for v in box_xyxy.tolist()]
    x1, y1 = max(x1,0), max(y1,0)
    x2, y2 = min(x2, gray01.shape[1]-1), min(y2, gray01.shape[0]-1)
    if x2 <= x1 or y2 <= y1: return 0.0
    patch = gray01[y1:y2, x1:x2]
    # count pixels darker than thresh (invert if your sketch is white lines on black)
    return float((patch < thresh).mean()) if patch.size > 0 else 0.0

def filter_boxes_sketch(
    boxes: np.ndarray,
    scores: np.ndarray,
    gray01: np.ndarray,
    *,
    max_area_frac: float = 0.40,
    min_density: float = 0.02,
    density_boost_over_global: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop boxes that are too large or too empty (few strokes)."""
    H, W = gray01.shape[:2]
    img_area = float(H * W)
    # global stroke density (dark %) — helps suppress 'full image' boxes
    global_density = float((gray01 < 0.8).mean())

    keep = []
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = b
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        area_frac = area / img_area if img_area > 0 else 1.0
        if area_frac > max_area_frac:
            continue
        d = stroke_density(gray01, b, thresh=0.8)
        # require some strokes + better than global (avoid whole-canvas boxes)
        if d < min_density: 
            continue
        if d < global_density + density_boost_over_global:
            continue
        keep.append(i)

    if not keep:
        return np.empty((0,4), np.float32), np.empty((0,), np.float32)
    keep = np.array(keep, dtype=np.int64)
    return boxes[keep], scores[keep]

# -------------- DINO core --------------
@torch.no_grad()
def detect_boxes_for_phrase(
    image_pil: Image.Image,
    phrase: str,
    processor,
    model,
    device: torch.device,
    *,
    text_threshold: float,
    box_threshold: float,
    nms_iou: float,
    max_boxes_per_label: int
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = image_pil.size
    inputs = processor(
        images=image_pil,
        text=[phrase],              # IMPORTANT: list
        padding="max_length",
        truncation=True,
        max_length=256,             # GroundingDINO text length
        return_tensors="pt"
    ).to(device)
    outputs = model(**inputs)
    processed = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=torch.tensor([[h, w]], device=device),
        input_ids=inputs["input_ids"],
        text_threshold=text_threshold
    )[0]
    boxes = processed["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = processed["scores"].detach().cpu().numpy().astype(np.float32)
    # confidence + NMS
    if len(boxes):
        keep = scores > box_threshold
        boxes, scores = boxes[keep], scores[keep]
    if len(boxes):
        keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), nms_iou).numpy()
        keep = keep[:max_boxes_per_label]
        boxes, scores = boxes[keep], scores[keep]
    return boxes, scores

def generate_tiles(img_pil: Image.Image, overlap: float = 0.1):
    """Yield (tile_pil, (x0,y0,x1,y1)) for a 3x3 grid (center+edges+corners)."""
    W, H = img_pil.size
    cols = rows = 3
    tile_w = int(round(W / cols))
    tile_h = int(round(H / rows))
    ox = int(round(tile_w * overlap))
    oy = int(round(tile_h * overlap))
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, c*tile_w - ox)
            y0 = max(0, r*tile_h - oy)
            x1 = min(W, (c+1)*tile_w + ox)
            y1 = min(H, (r+1)*tile_h + oy)
            yield img_pil.crop((x0,y0,x1,y1)), (x0,y0,x1,y1)

def remap_tile_boxes_to_full(tile_boxes: np.ndarray, tile_xyxy: Tuple[int,int,int,int]) -> np.ndarray:
    if len(tile_boxes)==0: return tile_boxes
    x0,y0,_,_ = tile_xyxy
    out = tile_boxes.copy()
    out[:,[0,2]] += x0
    out[:,[1,3]] += y0
    return out

# -------------- main --------------
def main():
    ap = argparse.ArgumentParser(description="Sketch-mode DINO test: full + tiles + sketch filters")
    ap.add_argument("--image", required=True)
    ap.add_argument("--components_json", required=True)
    ap.add_argument("--out_dir", default="dino_test")
    ap.add_argument("--grounding_model", default="IDEA-Research/grounding-dino-base")
    # thresholds
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.20)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--max_boxes_per_label", type=int, default=3)
    # sketch filters
    ap.add_argument("--max_area_frac", type=float, default=0.40)
    ap.add_argument("--min_density", type=float, default=0.02)
    ap.add_argument("--density_boost_over_global", type=float, default=0.25)
    # tiles
    ap.add_argument("--use_tiles", action="store_true", help="run also on a 3x3 grid and merge")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image
    image_pil = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_bgr.shape[:2]
    gray01 = to_gray_01(image_bgr)

    # comps
    components = load_components(args.components_json)
    if not components:
        raise ValueError("No components found in JSON.")

    # model
    processor = AutoProcessor.from_pretrained(args.grounding_model)
    try:
        processor.tokenizer.model_max_length = 256
    except Exception:
        pass
    model = GroundingDinoForObjectDetection.from_pretrained(args.grounding_model).to(device)
    model.eval()

    combined = image_bgr.copy()
    stats = {}
    missing = []

    for comp in components:
        phrase = comp.lower()

        # full image pass
        boxes_full, scores_full = detect_boxes_for_phrase(
            image_pil, phrase, processor, model, device,
            text_threshold=args.text_threshold,
            box_threshold=args.box_threshold,
            nms_iou=args.nms_iou,
            max_boxes_per_label=args.max_boxes_per_label
        )

        # optional tiles pass, then merge
        boxes_all = [boxes_full]
        scores_all = [scores_full]
        tiles_idx = []
        if args.use_tiles:
            for tile_pil, tile_xyxy in generate_tiles(image_pil, overlap=0.2):
                b_t, s_t = detect_boxes_for_phrase(
                    tile_pil, phrase, processor, model, device,
                    text_threshold=args.text_threshold,
                    box_threshold=args.box_threshold,
                    nms_iou=args.nms_iou,
                    max_boxes_per_label=2  # fewer per tile
                )
                b_t = remap_tile_boxes_to_full(b_t, tile_xyxy)
                boxes_all.append(b_t)
                scores_all.append(s_t)

        if len(boxes_all):
            boxes_cat = np.concatenate([b for b in boxes_all if len(b)], axis=0) if any(len(b) for b in boxes_all) else np.empty((0,4), np.float32)
            scores_cat = np.concatenate([s for s in scores_all if len(s)], axis=0) if any(len(s) for s in scores_all) else np.empty((0,), np.float32)
        else:
            boxes_cat = np.empty((0,4), np.float32)
            scores_cat = np.empty((0,), np.float32)

        # Global NMS to merge full + tiles
        if len(boxes_cat):
            keep = torchvision.ops.nms(torch.from_numpy(boxes_cat), torch.from_numpy(scores_cat), 0.5).numpy()
            boxes_cat, scores_cat = boxes_cat[keep], scores_cat[keep]

        # Sketch-mode filters: drop full-canvas, drop empty boxes
        boxes_filt, scores_filt = filter_boxes_sketch(
            boxes_cat, scores_cat, gray01,
            max_area_frac=args.max_area_frac,
            min_density=args.min_density,
            density_boost_over_global=args.density_boost_over_global
        )

        # Save per-component image
        slug = slugify(comp)
        vis_full = draw_boxes(image_bgr, boxes_filt, comp, scores=scores_filt, color=color_for(comp))
        cv2.imwrite(os.path.join(args.out_dir, f"{slug}_dino_sketch.png"), vis_full)

        # Add to combined
        combined = draw_boxes(combined, boxes_filt, comp, scores=scores_filt, color=color_for(comp))

        stats[comp] = {
            "boxes_xyxy": boxes_filt.tolist(),
            "scores": scores_filt.tolist(),
            "num_before_filters": int(len(boxes_cat)),
            "num_after_filters": int(len(boxes_filt)),
        }
        if len(boxes_filt) == 0:
            missing.append(comp)

    cv2.imwrite(os.path.join(args.out_dir, "dino_boxes_sketch.png"), combined)
    with open(os.path.join(args.out_dir, "dino_results_sketch.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out_dir, "missing_sketch.txt"), "w") as f:
        for m in missing:
            f.write(m + "\n")

    print(f"[OK] Saved {args.out_dir}/dino_boxes_sketch.png")
    print(f"[OK] Saved {args.out_dir}/dino_results_sketch.json")
    print(f"[OK] Missing after sketch-filters: {len(missing)} (see missing_sketch.txt)")
    if args.use_tiles:
        print("[INFO] Tiles were enabled (--use_tiles). Try varying overlap or thresholds if needed.")

if __name__ == "__main__":
    main()
