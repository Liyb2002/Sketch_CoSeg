#!/usr/bin/env python3
# file: run_grounded_sam2_components.py
import os, sys, json, re
from pathlib import Path
import numpy as np
import cv2
import torch

# --- edit these to your local checkpoints or adapt to your repo loaders ---
GROUNDING_DINO_CFG  = os.environ.get("GROUNDING_DINO_CFG",  "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CKPT = os.environ.get("GROUNDING_DINO_CKPT", "groundingdino_swint_ogc.pth")
SAM_CKPT            = os.environ.get("SAM_CKPT",            "sam_vit_h_4b8939.pth")  # or SAM2 ckpt if you use SAM2

BOX_THRESHOLD  = float(os.environ.get("BOX_THRESHOLD",  "0.25"))
TEXT_THRESHOLD = float(os.environ.get("TEXT_THRESHOLD", "0.20"))
NMS_IOU        = float(os.environ.get("NMS_IOU",        "0.50"))
MASK_THRESH    = float(os.environ.get("MASK_THRESH",    "0.50"))
TEXT_PREFIX    = os.environ.get("TEXT_PREFIX",          "motorcycle ")  # tweak or "" if not needed

# ------------------- utilities -------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_").lower()

def nms_numpy(boxes_xyxy, scores, iou_thr=0.5):
    if len(boxes_xyxy) == 0: return []
    boxes = np.asarray(boxes_xyxy, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2]-boxes[i, 0]) * (boxes[i, 3]-boxes[i, 1])
        area_o = (boxes[order[1:], 2]-boxes[order[1:], 0]) * (boxes[order[1:], 3]-boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter + 1e-8)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def save_mask_png(path: Path, mask_bool: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask_bool.astype(np.uint8) * 255))

# ------------------- model loading (adapt to your repo) -------------------
def load_grounding_dino(cfg_path, ckpt_path, device):
    # Grounded-SAM-2 repos usually provide a helper; here’s a common pattern:
    from groundingdino.util.inference import Model as GroundingDINOModel
    model = GroundingDINOModel(model_config_path=cfg_path, model_checkpoint_path=ckpt_path, device=device)
    return model

def grounding_dino_predict(model, image_bgr, text_prompt, box_thresh=0.25, text_thresh=0.20):
    # Returns (boxes_xyxy, scores)
    # GroundingDINO expects RGB float image
    from groundingdino.util import box_ops
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    boxes, logits, phrases = model.predict_with_caption(
        image=image_rgb,
        caption=text_prompt,
        box_threshold=box_thresh,
        text_threshold=text_thresh
    )
    # boxes are in absolute xyxy already in most helpers
    scores = logits.tolist() if hasattr(logits, "tolist") else [float(s) for s in logits]
    boxes_xyxy = boxes.tolist() if hasattr(boxes, "tolist") else boxes
    return boxes_xyxy, scores

def load_sam_predictor(sam_ckpt, device):
    # Use Meta’s SAM predictor (works fine inside Grounded-SAM-2 as well)
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)  # swap to SAM2 if needed
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def sam_predict_mask_from_box(predictor, image_bgr, box_xyxy, mask_thresh=0.5):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    box = np.array(box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=True)
    # Pick best by score, then threshold
    idx = int(np.argmax(scores))
    mask = masks[idx]  # bool already for SAM; if float, threshold
    if mask.dtype != np.bool_:
        mask = (mask >= mask_thresh)
    return mask, float(scores[idx])

# ------------------- main -------------------
def main():
    # script dir contains components.json and 0.png
    script_dir = Path(__file__).resolve().parent
    comp_json = script_dir / "components.json"
    img_path  = script_dir / "0.png"
    if not comp_json.exists():
        print(f"ERROR: {comp_json} not found", file=sys.stderr); sys.exit(1)
    if not img_path.exists():
        print(f"ERROR: {img_path} not found", file=sys.stderr); sys.exit(1)

    data = json.loads(comp_json.read_text())
    components = data.get("components", [])
    if not components:
        print("ERROR: components.json has no 'components' list.", file=sys.stderr); sys.exit(1)

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: failed to read {img_path}", file=sys.stderr); sys.exit(1)
    H, W = image.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    grounding = load_grounding_dino(GROUNDING_DINO_CFG, GROUNDING_DINO_CKPT, device)
    sam_pred  = load_sam_predictor(SAM_CKPT, device)

    # Outputs: masks saved in same folder, plus an annotations.json
    annotations = []
    instance_id = 1

    for name in components:
        label = (TEXT_PREFIX + name).strip()
        boxes_xyxy, scores = grounding_dino_predict(
            grounding, image, label, box_thresh=BOX_THRESHOLD, text_thresh=TEXT_THRESHOLD
        )
        # NMS then keep all remaining detections for that component
        keep = nms_numpy(boxes_xyxy, scores, iou_thr=NMS_IOU)
        boxes_xyxy = [boxes_xyxy[i] for i in keep]
        scores = [scores[i] for i in keep]

        if len(boxes_xyxy) == 0:
            print(f"[WARN] No detections for: {name}")

        for i, (box, sc) in enumerate(zip(boxes_xyxy, scores)):
            mask_bool, sam_sc = sam_predict_mask_from_box(sam_pred, image, box, mask_thresh=MASK_THRESH)
            # save mask png
            out_name = f"{sanitize(name)}_{i:03d}.png"
            out_path = script_dir / out_name
            save_mask_png(out_path, mask_bool)

            x1, y1, x2, y2 = [int(v) for v in box]
            ann = {
                "id": instance_id,
                "name": name,
                "file": out_name,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": int(mask_bool.sum()),
                "det_score": float(sc),
                "sam_score": float(sam_sc)
            }
            annotations.append(ann)
            instance_id += 1

    manifest = {
        "image": str(img_path.name),
        "components": components,
        "masks": annotations,
        "params": {
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "nms_iou": NMS_IOU,
            "mask_thresh": MASK_THRESH,
            "text_prefix": TEXT_PREFIX
        }
    }
    (script_dir / "annotations.json").write_text(json.dumps(manifest, indent=2))
    print("Done. Saved per-component masks and annotations.json in the same folder.")

if __name__ == "__main__":
    main()
