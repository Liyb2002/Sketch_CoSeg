#!/usr/bin/env python
# OWLv2 zero-shot detection with counts + global non-overlap.
# Also prints a summary and (when run directly) writes a visualization to owl_sam_out/.

import argparse
from typing import List, Dict, Tuple, Any
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

def _boxes_overlap(b1: np.ndarray, b2: np.ndarray) -> bool:
    x11,y11,x12,y12 = b1
    x21,y21,x22,y22 = b2
    iw = max(0.0, min(x12,x22) - max(x11,x21))
    ih = max(0.0, min(y12,y22) - max(y11,y21))
    return (iw > 0.0) and (ih > 0.0)

@torch.no_grad()
def _detect_once(image_pil: Image.Image, labels: List[str], model, processor, device):
    inputs = processor(text=labels, images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
    res = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
    boxes  = res["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = res["scores"].detach().cpu().numpy().astype(np.float32)
    lids   = res["labels"].detach().cpu().numpy().astype(np.int64)
    return boxes, scores, lids

def _pool_candidates(labels: List[str], per_boxes, per_scores, per_lids) -> Dict[str, Tuple[np.ndarray,np.ndarray]]:
    pooled = {u: ([],[]) for u in labels}
    for B, S, L in zip(per_boxes, per_scores, per_lids):
        for i, u in enumerate(labels):
            sel = (L == i)
            if np.any(sel):
                pooled[u][0].append(B[sel]); pooled[u][1].append(S[sel])
    out: Dict[str, Tuple[np.ndarray,np.ndarray]] = {}
    for u in labels:
        if pooled[u][0]:
            out[u] = (np.concatenate(pooled[u][0], 0), np.concatenate(pooled[u][1], 0))
        else:
            out[u] = (np.empty((0,4), np.float32), np.empty((0,), np.float32))
    return out

def _nms_topk(boxes: np.ndarray, scores: np.ndarray, iou: float, k: int):
    if len(boxes) == 0: return boxes, scores
    idx = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    idx = idx[:k]
    return boxes[idx], scores[idx]

def _select_counts_no_overlap(
    items: List[Dict[str, Any]],
    pooled: Dict[str, Tuple[np.ndarray,np.ndarray]],
    nms_iou: float,
    score_thresholds: List[float],
    enforce_no_overlap: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Greedy global selection:
      1) For each label, threshold sweep -> NMS -> candidate pool.
      2) Merge all candidates across labels, sort by score desc.
      3) Pick greedily if:
            - that label still needs more boxes
            - and (optionally) it does not overlap any already selected box.
    """
    per_label_cands: Dict[str, Tuple[np.ndarray,np.ndarray]] = {}
    for name, (B, S) in pooled.items():
        used_thr = None
        B_ord, S_ord = B[np.argsort(-S)], S[np.argsort(-S)]
        chosenB = np.empty((0,4), np.float32); chosenS = np.empty((0,), np.float32)
        for thr in score_thresholds:
            keep = S_ord >= thr
            b, s = B_ord[keep], S_ord[keep]
            if len(b):
                b2, s2 = _nms_topk(b, s, nms_iou, k=len(b))
                if len(b2):
                    chosenB, chosenS, used_thr = b2, s2, thr
                    break
        if used_thr is None:
            chosenB, chosenS, used_thr = np.empty((0,4), np.float32), np.empty((0,), np.float32), score_thresholds[-1]
        per_label_cands[name] = (chosenB, chosenS)

    # Build global candidate list
    all_cands = []
    for name, (B, S) in per_label_cands.items():
        for i in range(len(B)):
            all_cands.append((float(S[i]), name, B[i]))
    all_cands.sort(key=lambda x: -x[0])  # score desc

    # Greedy with non-overlap
    need = {it["name"]: int(it["count"]) for it in items}
    picked: Dict[str, List[np.ndarray]] = {it["name"]: [] for it in items}
    picked_scores: Dict[str, List[float]] = {it["name"]: [] for it in items}
    global_boxes: List[np.ndarray] = []

    for sc, nm, bx in all_cands:
        if need.get(nm, 0) <= 0:  # already satisfied
            continue
        ok = True
        if enforce_no_overlap:
            for gb in global_boxes:
                if _boxes_overlap(bx, gb):
                    ok = False
                    break
        if not ok:
            continue
        # accept
        picked[nm].append(bx)
        picked_scores[nm].append(sc)
        global_boxes.append(bx)
        need[nm] -= 1

    # Package results
    out = {}
    for it in items:
        nm = it["name"]
        if picked[nm]:
            out[nm] = {
                "boxes": np.stack(picked[nm], 0).astype(np.float32),
                "scores": np.array(picked_scores[nm], np.float32),
                "thr": None
            }
        else:
            out[nm] = {"boxes": np.empty((0,4), np.float32), "scores": np.empty((0,), np.float32), "thr": None}
    return out

def detect_owlv2_boxes_counts(
    image_pil: Image.Image,
    items: List[Dict[str, Any]],              # [{name, count}, ...]
    model_id: str = "google/owlv2-large-patch14",
    use_tiles: bool = True,
    tile_grid: int = 3,
    tile_overlap: float = 0.2,
    nms_iou: float = 0.5,
    score_thresholds: List[float] = (0.30,0.25,0.20,0.15,0.10,0.07,0.05,0.03,0.01,0.0),
    enforce_no_overlap: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    labels = [it["name"] for it in items]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        torch_dtype=(torch.float16 if torch.cuda.is_available() else None),
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # full image
    b_full, s_full, lid_full = _detect_once(image_pil, labels, model, processor, device)
    per_boxes = [b_full]; per_scores = [s_full]; per_lids = [lid_full]

    # tiles
    if use_tiles:
        for tile_pil, tile_xyxy in _gen_tiles(image_pil, tile_grid, tile_overlap):
            b, s, lid = _detect_once(tile_pil, labels, model, processor, device)
            b = _remap_boxes(b, tile_xyxy)
            per_boxes.append(b); per_scores.append(s); per_lids.append(lid)

    pooled = _pool_candidates(labels, per_boxes, per_scores, per_lids)
    results = _select_counts_no_overlap(items, pooled, nms_iou, list(score_thresholds), enforce_no_overlap)

    # ---- summary print ----
    total = len(items)
    ok_names = [nm for nm, v in results.items() if len(v["boxes"]) >= 1]
    miss_names = [it["name"] for it in items if len(results[it["name"]]["boxes"]) == 0]
    print(f"[OWL] detected {len(ok_names)} / {total} components with >=1 box.")
    if miss_names:
        print("[OWL] not detected:", ", ".join(miss_names))

    return results

# ------------- CLI: writes visualization -------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--components_json", required=True)
    ap.add_argument("--out_dir", default="./owl_sam_out")
    ap.add_argument("--model_id", default="google/owlv2-large-patch14")
    ap.add_argument("--use_tiles", action="store_true")
    ap.add_argument("--tile_grid", type=int, default=3)
    ap.add_argument("--tile_overlap", type=float, default=0.2)
    args = ap.parse_args()

    import json, cv2
    from pathlib import Path

    payload = json.load(open(args.components_json))
    comps = payload.get("components", [])
    items = []
    if comps and isinstance(comps[0], dict):
        for it in comps:
            n = str(it.get("name","")).strip()
            c = int(it.get("count", 1))
            if n and c >= 1: items.append({"name": n, "count": c})
    else:
        for n in comps:
            if isinstance(n, str) and n.strip():
                items.append({"name": n.strip(), "count": 1})

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    res = detect_owlv2_boxes_counts(
        image_pil=img, items=items, model_id=args.model_id,
        use_tiles=args.use_tiles, tile_grid=args.tile_grid, tile_overlap=args.tile_overlap,
        enforce_no_overlap=True
    )

    # visualization
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    def color_for(lbl: str) -> tuple[int,int,int]:
        rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
        c = rng.integers(50, 220, size=3, dtype=np.int32)
        return (int(c[0]), int(c[1]), int(c[2]))
    for it in items:
        nm = it["name"]; col = color_for(nm)
        boxes = res[nm]["boxes"]; scores = res[nm]["scores"]
        for i, b in enumerate(boxes):
            x1,y1,x2,y2 = [int(round(v)) for v in b.tolist()]
            cv2.rectangle(bgr, (x1,y1), (x2,y2), col, 2)
            txt = f"{nm}:{scores[i]:.2f}" if i < len(scores) else nm
            cv2.putText(bgr, txt, (x1, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)
    out_path = str(Path(args.out_dir) / "owl_boxes.png")
    cv2.imwrite(out_path, bgr)
    print(f"[OK] wrote {out_path}")
