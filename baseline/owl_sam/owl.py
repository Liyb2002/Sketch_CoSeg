#!/usr/bin/env python
# OWLv2 zero-shot detection helper (optionally tiled), with a tiny CLI.

import argparse
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torchvision
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def _remap_boxes(tile_boxes: np.ndarray, tile_xyxy: Tuple[int,int,int,int]) -> np.ndarray:
    if len(tile_boxes) == 0: return tile_boxes
    x0,y0,_,_ = tile_xyxy
    out = tile_boxes.copy()
    out[:,[0,2]] += x0; out[:,[1,3]] += y0
    return out

def _gen_tiles(pil_img: Image.Image, grid: int, overlap: float):
    W, H = pil_img.size
    tw, th = int(round(W / grid)), int(round(H / grid))
    ox, oy = int(round(tw * overlap)), int(round(th * overlap))
    for r in range(grid):
        for c in range(grid):
            x0 = max(0, c*tw - ox); y0 = max(0, r*th - oy)
            x1 = min(W, (c+1)*tw + ox); y1 = min(H, (r+1)*th + oy)
            yield pil_img.crop((x0,y0,x1,y1)), (x0,y0,x1,y1)

def _nms_topk(boxes: np.ndarray, scores: np.ndarray, iou: float, k: int):
    if len(boxes) == 0: return boxes, scores
    idx = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    idx = idx[:k]
    return boxes[idx], scores[idx]

@torch.no_grad()
def _detect_once(image_pil: Image.Image, labels: List[str], model, processor, device):
    inputs = processor(text=labels, images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
    res = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
    return (res["boxes"].detach().cpu().numpy().astype(np.float32),
            res["scores"].detach().cpu().numpy().astype(np.float32),
            res["labels"].detach().cpu().numpy().astype(np.int64))

def detect_owlv2_boxes(
    image_pil: Image.Image,
    labels: List[str],
    model_id: str = "google/owlv2-large-patch14",
    use_tiles: bool = True,
    tile_grid: int = 3,
    tile_overlap: float = 0.2,
    nms_iou: float = 0.5,
    score_thresholds: List[float] = (0.30, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01, 0.0),
    max_per_label: int = 5,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns: dict[label] = {"boxes": (K,4) float32, "scores": (K,), "thr": float}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, dtype=(torch.float16 if torch.cuda.is_available() else None)
    ).to(device).eval()

    # full image pass
    b_full, s_full, lid_full = _detect_once(image_pil, labels, model, processor, device)
    per_boxes = [b_full]; per_scores = [s_full]; per_lids = [lid_full]

    # optional tiles
    if use_tiles:
        for tile_pil, tile_xyxy in _gen_tiles(image_pil, tile_grid, tile_overlap):
            b, s, lid = _detect_once(tile_pil, labels, model, processor, device)
            b = _remap_boxes(b, tile_xyxy)
            per_boxes.append(b); per_scores.append(s); per_lids.append(lid)

    # pool by label id
    pooled = {u: ([],[]) for u in labels}
    for boxes, scores, lids in zip(per_boxes, per_scores, per_lids):
        for i, u in enumerate(labels):
            sel = (lids == i)
            if np.any(sel):
                pooled[u][0].append(boxes[sel]); pooled[u][1].append(scores[sel])

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for u in labels:
        if pooled[u][0]:
            B = np.concatenate(pooled[u][0], 0); S = np.concatenate(pooled[u][1], 0)
        else:
            B = np.empty((0,4), np.float32); S = np.empty((0,), np.float32)

        # threshold sweep + NMS; keep up to max_per_label
        used_thr = None
        chosen_b = np.empty((0,4), np.float32); chosen_s = np.empty((0,), np.float32)
        order = np.argsort(-S); B, S = B[order], S[order]
        for thr in score_thresholds:
            keep = S >= thr
            b, s = B[keep], S[keep]
            if len(b):
                b2, s2 = _nms_topk(b, s, nms_iou, max_per_label)
                if len(b2):
                    chosen_b, chosen_s = b2, s2
                    used_thr = thr
                    break
        if used_thr is None:
            used_thr = score_thresholds[-1]
        out[u] = {"boxes": chosen_b, "scores": chosen_s, "thr": used_thr}
    return out

# ---------- simple CLI test ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="owl_debug.png")
    ap.add_argument("--model_id", default="google/owlv2-large-patch14")
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    det = detect_owlv2_boxes(img, args.labels, model_id=args.model_id)

    # quick visualization
    import cv2
    import numpy as np
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    def color_for(lbl: str) -> tuple[int,int,int]:
        rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
        c = rng.integers(50, 220, size=3, dtype=np.int32)
        return (int(c[0]), int(c[1]), int(c[2]))
    for lbl in args.labels:
        boxes = det[lbl]["boxes"]; scores = det[lbl]["scores"]
        col = color_for(lbl)
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = [int(round(v)) for v in b.tolist()]
            cv2.rectangle(bgr, (x1,y1), (x2,y2), col, 2)
            txt = f"{lbl}:{scores[i]:.2f}" if i < len(scores) else lbl
            cv2.putText(bgr, txt, (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
    cv2.imwrite(args.out, bgr)
    print(f"[OK] saved {args.out}")
