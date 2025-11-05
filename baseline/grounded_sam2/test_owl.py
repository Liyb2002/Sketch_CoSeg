#!/usr/bin/env python
import argparse, json, os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
from PIL import Image
import cv2
import torchvision
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# ------------- utils -------------
def ensure_dir(p: str | Path): Path(p).mkdir(parents=True, exist_ok=True)
def slugify(s: str) -> str: return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")
def color_for(lbl: str) -> tuple[int,int,int]:
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))  # BGR

def draw_boxes(img_bgr: np.ndarray, boxes_xyxy: np.ndarray, label: str,
               scores: np.ndarray | None = None, color: tuple[int,int,int] | None = None,
               thickness: int = 2) -> np.ndarray:
    vis = img_bgr.copy()
    color = color or color_for(label)
    for i, b in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        txt = label if scores is None or i >= len(scores) else f"{label}: {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ytxt = max(0, y1 - 6)
        cv2.rectangle(vis, (x1, ytxt - th - 4), (x1 + tw + 6, ytxt), (0, 0, 0), -1)
        cv2.putText(vis, txt, (x1 + 3, ytxt - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
    return vis

def load_items(json_path: str) -> List[dict]:
    payload = json.load(open(json_path))
    items = []
    for it in payload.get("components", []):
        if isinstance(it, dict) and "name" in it and "count" in it:
            name = str(it["name"]).strip()
            try:
                cnt = int(it["count"])
            except Exception:
                continue
            if name and cnt > 0:
                items.append({"name": name, "count": cnt})
    if not items:
        raise ValueError("No valid components found in JSON.")
    return items

# ------------- tiling -------------
def gen_tiles(pil_img: Image.Image, grid: int = 3, overlap: float = 0.2):
    W, H = pil_img.size
    tw, th = int(round(W / grid)), int(round(H / grid))
    ox, oy = int(round(tw * overlap)), int(round(th * overlap))
    for r in range(grid):
        for c in range(grid):
            x0 = max(0, c*tw - ox); y0 = max(0, r*th - oy)
            x1 = min(W, (c+1)*tw + ox); y1 = min(H, (r+1)*th + oy)
            yield pil_img.crop((x0,y0,x1,y1)), (x0,y0,x1,y1)

def remap_boxes(tile_boxes: np.ndarray, tile_xyxy: Tuple[int,int,int,int]) -> np.ndarray:
    if len(tile_boxes) == 0: return tile_boxes
    x0,y0,_,_ = tile_xyxy
    out = tile_boxes.copy()
    out[:,[0,2]] += x0; out[:,[1,3]] += y0
    return out

# ------------- OWL-ViT -------------
@torch.no_grad()
def owlvit_detect_once(
    image_pil: Image.Image,
    labels: List[str],
    model: OwlViTForObjectDetection,
    processor: OwlViTProcessor,
    device: torch.device,
):
    inputs = processor(text=labels, images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)  # (H, W)
    res = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
    return (res["boxes"].detach().cpu().numpy().astype(np.float32),
            res["scores"].detach().cpu().numpy().astype(np.float32),
            res["labels"].detach().cpu().numpy().astype(np.int64))

def pool_per_label(unique_labels: List[str],
                   per_pass_boxes: List[np.ndarray],
                   per_pass_scores: List[np.ndarray],
                   per_pass_label_ids: List[np.ndarray]) -> Dict[str, Tuple[np.ndarray,np.ndarray]]:
    pooled = {u: ([],[]) for u in unique_labels}
    for boxes, scores, label_ids in zip(per_pass_boxes, per_pass_scores, per_pass_label_ids):
        for i, u in enumerate(unique_labels):
            sel = (label_ids == i)
            if np.any(sel):
                pooled[u][0].append(boxes[sel])
                pooled[u][1].append(scores[sel])
    out = {}
    for u in unique_labels:
        bparts, sparts = pooled[u]
        if bparts:
            out[u] = (np.concatenate(bparts, 0), np.concatenate(sparts, 0))
        else:
            out[u] = (np.empty((0,4), np.float32), np.empty((0,), np.float32))
    return out

def nms_topk(boxes: np.ndarray, scores: np.ndarray, iou: float, k: int):
    if len(boxes) == 0: return boxes, scores
    idx = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    idx = idx[:k]
    return boxes[idx], scores[idx]

def try_fill_count_with_threshold_sweep(
    boxes: np.ndarray, scores: np.ndarray, need: int, nms_iou: float,
    sweeps=(0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01, 0.0)
) -> Tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    for thr in sweeps:
        keep = scores >= thr
        b, s = boxes[keep], scores[keep]
        if len(b):
            b2, s2 = nms_topk(b, s, nms_iou, need)
            if len(b2) >= need:
                return b2[:need], s2[:need], thr
    # fallback: whatever we have after NMS on loosest thr
    b2, s2 = nms_topk(boxes[scores >= sweeps[-1]], scores[scores >= sweeps[-1]], nms_iou, need)
    return b2, s2, sweeps[-1]

def jitter_duplicates(box: np.ndarray, how_many: int, W: int, H: int) -> np.ndarray:
    x1,y1,x2,y2 = box.tolist()
    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
    out = [box]
    for _ in range(how_many - 1):
        dx = (np.random.rand() - 0.5) * 0.10 * w  # ±5%
        dy = (np.random.rand() - 0.5) * 0.10 * h
        jx1 = np.clip(x1 + dx, 0, W - 1); jy1 = np.clip(y1 + dy, 0, H - 1)
        jx2 = np.clip(x2 + dx, 0, W - 1); jy2 = np.clip(y2 + dy, 0, H - 1)
        if jx2 <= jx1: jx2 = min(W - 1, jx1 + 1)
        if jy2 <= jy1: jy2 = min(H - 1, jy1 + 1)
        out.append(np.array([jx1, jy1, jx2, jy2], dtype=np.float32))
    return np.stack(out, 0)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="OWL-ViT counted boxes (exact count per component)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--components_json", required=True)
    ap.add_argument("--out_dir", default="owl_test")

    # default to higher-accuracy model (smaller patch, larger backbone)
    ap.add_argument("--model_id", default="google/owlvit-large-patch14")

    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--use_tiles", action="store_true")
    ap.add_argument("--tile_grid", type=int, default=3)
    ap.add_argument("--tile_overlap", type=float, default=0.2)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # image
    image_pil = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_rgb.shape[:2]

    # items & unique label list
    items = load_items(args.components_json)
    unique_labels = []
    for it in items:
        if it["name"] not in unique_labels:
            unique_labels.append(it["name"])

    # model
    processor = OwlViTProcessor.from_pretrained(args.model_id)
    model = OwlViTForObjectDetection.from_pretrained(args.model_id).to(device).eval()

    # full image
    boxes_full, scores_full, lid_full = owlvit_detect_once(image_pil, unique_labels, model, processor, device)
    all_boxes = [boxes_full]; all_scores = [scores_full]; all_lids = [lid_full]

    # optional tiles
    if args.use_tiles:
        for tile_pil, tile_xyxy in gen_tiles(image_pil, grid=args.tile_grid, overlap=args.tile_overlap):
            b, s, lid = owlvit_detect_once(tile_pil, unique_labels, model, processor, device)
            b = remap_boxes(b, tile_xyxy)
            all_boxes.append(b); all_scores.append(s); all_lids.append(lid)

    pooled = pool_per_label(unique_labels, all_boxes, all_scores, all_lids)

    # select EXACT counts per item
    results = []
    combined = image_bgr.copy()
    fallback_notes = []

    for it in items:
        name, count = it["name"], int(it["count"])
        cand_boxes, cand_scores = pooled.get(name, (np.empty((0,4), np.float32), np.empty((0,), np.float32)))

        chosen_boxes, chosen_scores, used_thr = try_fill_count_with_threshold_sweep(
            cand_boxes, cand_scores, need=count, nms_iou=args.nms_iou,
            sweeps=(0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01, 0.0)
        )

        fallback = False
        if len(chosen_boxes) < count:
            if len(chosen_boxes) >= 1:
                dup = jitter_duplicates(chosen_boxes[0], count - len(chosen_boxes) + 1, W, H)[1:]
                chosen_boxes = np.concatenate([chosen_boxes, dup], 0)[:count]
                chosen_scores = np.concatenate([chosen_scores, np.zeros((len(dup),), np.float32)], 0)[:count]
                fallback = True
            else:
                # total fallback: center proposals
                cx, cy = W/2, H/2
                bw, bh = 0.1*W, 0.1*H
                base = np.array([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], np.float32)
                chosen_boxes = jitter_duplicates(base, count, W, H)
                chosen_scores = np.zeros((count,), np.float32)
                fallback = True

        slug = slugify(name)
        vis = draw_boxes(image_bgr, chosen_boxes, name, scores=chosen_scores, color=color_for(name))
        cv2.imwrite(os.path.join(args.out_dir, f"{slug}_owl.png"), vis)
        combined = draw_boxes(combined, chosen_boxes, name, scores=chosen_scores, color=color_for(name))

        if fallback:
            fallback_notes.append(f"{name} (requested {count}, fallback used)")

        results.append({
            "name": name,
            "count": count,
            "boxes_xyxy": chosen_boxes.tolist(),
            "scores": chosen_scores.tolist(),
            "used_threshold": float(used_thr),
            "fallback": bool(fallback)
        })

    cv2.imwrite(os.path.join(args.out_dir, "owl_boxes.png"), combined)
    json.dump({"items": results, "unique_labels": unique_labels}, open(os.path.join(args.out_dir, "owl_results.json"), "w"), indent=2)
    with open(os.path.join(args.out_dir, "missing.txt"), "w") as f:
        for line in fallback_notes: f.write(line + "\n")

    print(f"[OK] Saved: {args.out_dir}/owl_boxes.png")
    print(f"[OK] Saved: {args.out_dir}/owl_results.json")
    if fallback_notes:
        print(f"[WARN] Fallback used for {len(fallback_notes)} item(s) (see missing.txt)")

if __name__ == "__main__":
    main()
