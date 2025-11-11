#!/usr/bin/env python
# OWLv2 zero-shot detection with counts + global non-overlap.
# Supports:
#   - optional per-item "query" (e.g. "door of a car"),
#   - tiling for large images,
#   - greedy global selection with per-item counts and optional non-overlap,
#   - visualization via CLI.
#
# NEW:
#   - By default, saves all final boxes + scores to "owl_boxes.json".
#   - Callers may override via the `save_json` argument.

import argparse
import json
from typing import List, Dict, Tuple, Any
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ------------- global OWL cache -------------
_OWL_CACHE = None  # (model_id, device_str, processor, model)


def _get_owl_model(model_id: str, device: torch.device):
    """
    Cache OWL model+processor per (model_id, device) to avoid reloads.
    """
    global _OWL_CACHE
    key = (model_id, str(device))

    if _OWL_CACHE is not None and _OWL_CACHE[0:2] == key:
        return _OWL_CACHE[2], _OWL_CACHE[3]

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id,
        torch_dtype=(torch.float16 if device.type == "cuda" else None),
        low_cpu_mem_usage=True,
    ).to(device).eval()

    _OWL_CACHE = (model_id, str(device), processor, model)
    return processor, model


# ------------- helpers -------------
def _remap_boxes(tile_boxes: np.ndarray, tile_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    if len(tile_boxes) == 0:
        return tile_boxes
    x0, y0, _, _ = tile_xyxy
    out = tile_boxes.copy()
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out


def _gen_tiles(pil_img: Image.Image, grid: int, overlap: float):
    W, H = pil_img.size
    tw, th = int(round(W / grid)), int(round(H / grid))
    ox, oy = int(round(tw * overlap)), int(round(th * overlap))
    for r in range(grid):
        for c in range(grid):
            x0 = max(0, c * tw - ox)
            y0 = max(0, r * th - oy)
            x1 = min(W, (c + 1) * tw + ox)
            y1 = min(H, (r + 1) * th + oy)
            yield pil_img.crop((x0, y0, x1, y1)), (x0, y0, x1, y1)


def _boxes_overlap(b1: np.ndarray, b2: np.ndarray) -> bool:
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    iw = max(0.0, min(x12, x22) - max(x11, x21))
    ih = max(0.0, min(y12, y22) - max(y11, y21))
    return (iw > 0.0) and (ih > 0.0)


@torch.no_grad()
def _detect_once(image_pil: Image.Image, label_texts: List[str], model, processor, device):
    """
    Run OWLv2 once for given label texts.
    """
    inputs = processor(text=label_texts, images=image_pil, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image_pil.size[::-1]], device=device)
    res = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]

    boxes = res["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = res["scores"].detach().cpu().numpy().astype(np.float32)
    lids = res["labels"].detach().cpu().numpy().astype(np.int64)
    return boxes, scores, lids


def _pool_candidates(
    label_names: List[str],
    per_boxes: List[np.ndarray],
    per_scores: List[np.ndarray],
    per_lids: List[np.ndarray],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Pool boxes/scores from full + tiles by label index.
    label_names: order corresponds to class indices used in OWL.
    """
    pooled = {nm: ([], []) for nm in label_names}

    for B, S, L in zip(per_boxes, per_scores, per_lids):
        for i, nm in enumerate(label_names):
            sel = (L == i)
            if np.any(sel):
                pooled[nm][0].append(B[sel])
                pooled[nm][1].append(S[sel])

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for nm in label_names:
        if pooled[nm][0]:
            out[nm] = (
                np.concatenate(pooled[nm][0], axis=0),
                np.concatenate(pooled[nm][1], axis=0),
            )
        else:
            out[nm] = (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
    return out


def _nms_topk(boxes: np.ndarray, scores: np.ndarray, iou: float, k: int):
    if len(boxes) == 0:
        return boxes, scores
    keep = torchvision.ops.nms(
        torch.from_numpy(boxes),
        torch.from_numpy(scores),
        iou,
    ).numpy()
    keep = keep[:k]
    return boxes[keep], scores[keep]


def _select_counts_no_overlap(
    items: List[Dict[str, Any]],
    pooled: Dict[str, Tuple[np.ndarray, np.ndarray]],
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
    # 1) Per-label candidates
    per_label_cands: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, (B, S) in pooled.items():
        B_ord = B[np.argsort(-S)]
        S_ord = S[np.argsort(-S)]
        chosenB = np.empty((0, 4), dtype=np.float32)
        chosenS = np.empty((0,), dtype=np.float32)

        for thr in score_thresholds:
            keep = (S_ord >= thr)
            b, s = B_ord[keep], S_ord[keep]
            if len(b):
                b2, s2 = _nms_topk(b, s, nms_iou, k=len(b))
                if len(b2):
                    chosenB, chosenS = b2, s2
                    break

        per_label_cands[name] = (chosenB, chosenS)

    # 2) Global candidate list
    all_cands = []
    for name, (B, S) in per_label_cands.items():
        for i in range(len(B)):
            all_cands.append((float(S[i]), name, B[i]))
    all_cands.sort(key=lambda x: -x[0])

    # 3) Greedy selection with per-item counts
    need = {it["name"]: int(it.get("count", 1)) for it in items}
    picked: Dict[str, List[np.ndarray]] = {it["name"]: [] for it in items}
    picked_scores: Dict[str, List[float]] = {it["name"]: [] for it in items}
    global_boxes: List[np.ndarray] = []

    for sc, nm, bx in all_cands:
        if need.get(nm, 0) <= 0:
            continue

        ok = True
        if enforce_no_overlap:
            for gb in global_boxes:
                if _boxes_overlap(bx, gb):
                    ok = False
                    break
        if not ok:
            continue

        picked[nm].append(bx)
        picked_scores[nm].append(sc)
        global_boxes.append(bx)
        need[nm] -= 1

    # 4) Package results
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for it in items:
        nm = it["name"]
        if picked[nm]:
            out[nm] = {
                "boxes": np.stack(picked[nm], axis=0).astype(np.float32),
                "scores": np.array(picked_scores[nm], dtype=np.float32),
                "thr": None,
            }
        else:
            out[nm] = {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "thr": None,
            }
    return out


def _save_boxes_json(results: Dict[str, Dict[str, np.ndarray]], save_path: str):
    """
    Save OWL results (boxes + scores) as JSON.

    Schema:
    {
      "labels": {
        "<label>": {
          "boxes": [[x1,y1,x2,y2], ...],
          "scores": [s1, s2, ...]
        },
        ...
      }
    }
    """
    serializable = {"labels": {}}
    for lbl, v in results.items():
        boxes = v.get("boxes", np.empty((0, 4), np.float32))
        scores = v.get("scores", np.empty((0,), np.float32))
        serializable["labels"][lbl] = {
            "boxes": boxes.tolist(),
            "scores": scores.tolist(),
        }

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(serializable, f, indent=2)


# ------------- public API -------------
def detect_owlv2_boxes_counts(
    image_pil: Image.Image,
    items: List[Dict[str, Any]],              # [{name, count, (optional) query}, ...]
    model_id: str = "google/owlv2-large-patch14",
    use_tiles: bool = True,
    tile_grid: int = 3,
    tile_overlap: float = 0.2,
    nms_iou: float = 0.5,
    score_thresholds: List[float] = (0.30,0.25,0.20,0.15,0.10,0.07,0.05,0.03,0.01,0.0),
    enforce_no_overlap: bool = False,
    save_json: str = None,                    # if None -> defaults to "./owl_boxes.json"
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run OWLv2 detection with global per-item count selection.

    - If an item has "query", that is used as the OWL text prompt.
      Otherwise item["name"] is used.
    - Results are keyed by plain item["name"].
    - Always saves a JSON with the final boxes and scores:
        - if save_json is provided, use that path
        - else write to "owl_boxes.json" in the current working directory.
    """

    # --- build unique label texts + mapping to names ---
    label_names: List[str] = []
    label_texts: List[str] = []
    for it in items:
        nm = str(it.get("name", "")).strip()
        if not nm:
            continue
        if nm not in label_names:
            label_names.append(nm)
            q = str(it.get("query") or nm)
            label_texts.append(q)

    if not label_names:
        return {}

    # --- model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = _get_owl_model(model_id, device)

    # --- full image ---
    b_full, s_full, lid_full = _detect_once(image_pil, label_texts, model, processor, device)
    per_boxes = [b_full]
    per_scores = [s_full]
    per_lids = [lid_full]

    # --- tiles (optional) ---
    if use_tiles:
        for tile_pil, tile_xyxy in _gen_tiles(image_pil, tile_grid, tile_overlap):
            b, s, lid = _detect_once(tile_pil, label_texts, model, processor, device)
            per_boxes.append(_remap_boxes(b, tile_xyxy))
            per_scores.append(s)
            per_lids.append(lid)

    # --- pool strictly by class index -> label_names ---
    pooled = _pool_candidates(label_names, per_boxes, per_scores, per_lids)

    # --- per-item selection ---
    results = _select_counts_no_overlap(
        items=items,
        pooled=pooled,
        nms_iou=nms_iou,
        score_thresholds=list(score_thresholds),
        enforce_no_overlap=enforce_no_overlap,
    )

    # --- tiny summary (unchanged behavior) ---
    total = len(items)
    ok_names = [nm for nm, v in results.items() if len(v["boxes"]) >= 1]
    miss_names = [it["name"] for it in items if len(results[it["name"]]["boxes"]) == 0]
    print(f"[OWL] detected {len(ok_names)} / {total} components with >=1 box.")
    if miss_names:
        print("[OWL] not detected:", ", ".join(miss_names))

    return results

# ------------- CLI: visualize + save JSON in out_dir -------------
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.load(open(args.components_json))
    comps = payload.get("components", [])
    items: List[Dict[str, Any]] = []
    if comps and isinstance(comps, list) and comps and isinstance(comps[0], dict):
        for it in comps:
            n = str(it.get("name", "")).strip()
            c = int(it.get("count", 1))
            if n and c >= 1:
                items.append({"name": n, "count": c})
    else:
        for n in comps:
            if isinstance(n, str) and n.strip():
                items.append({"name": n.strip(), "count": 1})

    img = Image.open(args.image).convert("RGB")

    # For CLI we explicitly save JSON inside out_dir.
    boxes_json_path = str(out_dir / "owl_boxes.json")
    res = detect_owlv2_boxes_counts(
        image_pil=img,
        items=items,
        model_id=args.model_id,
        use_tiles=args.use_tiles,
        tile_grid=args.tile_grid,
        tile_overlap=args.tile_overlap,
        save_json=boxes_json_path,
    )

    # Simple visualization for debugging
    import cv2
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _vis_color(lbl: str) -> Tuple[int, int, int]:
        rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
        c = rng.integers(50, 220, size=3, dtype=np.int32)
        return int(c[0]), int(c[1]), int(c[2])

    for it in items:
        nm = it["name"]
        col = _vis_color(nm)
        boxes = res[nm]["boxes"]
        scores = res[nm]["scores"]
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
            if i < len(scores):
                txt = f"{nm}:{scores[i]:.2f}"
            else:
                txt = nm
            cv2.putText(
                bgr,
                txt,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    vis_path = out_dir / "owl_boxes.png"
    cv2.imwrite(str(vis_path), bgr)
    print(f"[OK] wrote {vis_path}")
    print(f"[OK] wrote {boxes_json_path}")
