#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import cv2
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from segment_anything import sam_model_registry, SamPredictor
import torchvision


# -------------------------------
# Utility
# -------------------------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_mask_png(mask, out_path):
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(out_path, mask_u8)


def save_overlay(image_bgr, masks, labels, out_path):
    # Start in float for blending, then convert to uint8 for cv2 drawing
    overlay = image_bgr.copy().astype(np.float32)

    def color_for(lbl):
        rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
        # Return a plain Python int tuple (B, G, R)
        c = rng.integers(50, 220, size=3, dtype=np.int32)
        return (int(c[0]), int(c[1]), int(c[2]))

    # Blend each mask
    for m, lbl in zip(masks, labels):
        if m is None:
            continue
        mask2 = m.astype(bool)                  # HxW
        col = np.array(color_for(lbl), np.float32)  # (3,)
        # overlay[mask2] is (N,3); broadcast col
        overlay[mask2] = 0.6 * overlay[mask2] + 0.4 * col

    # Convert to uint8 before cv2 drawing
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Legend
    y = 24
    for lbl in labels:
        col = color_for(lbl)                    # (B,G,R) ints
        cv2.rectangle(overlay, (10, int(y - 15)), (30, int(y + 5)), col, thickness=-1)
        cv2.putText(overlay, lbl, (40, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22

    cv2.imwrite(out_path, overlay)


# -------------------------------
# Loaders
# -------------------------------
def load_components(json_path):
    with open(json_path, "r") as f:
        payload = json.load(f)
    comps = payload.get("components", [])
    return [c.strip() for c in comps if isinstance(c, str) and c.strip()]


def load_grounding_dino(model_id, device):
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        processor.tokenizer.model_max_length = 256
    except Exception:
        pass
    model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


def load_sam(sam_type, sam_ckpt, device):
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


# -------------------------------
# Grounding DINO inference
# -------------------------------
@torch.no_grad()
def dino_detect_boxes_for_phrase(
    image_pil,
    phrase,
    processor,
    model,
    device,
    box_threshold=0.25,
    text_threshold=0.20,
):
    w, h = image_pil.size

    inputs = processor(
        images=image_pil,
        text=[phrase],
        padding="max_length",
        truncation=True,
        max_length=256,  # ensure same as model text encoder
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)

    processed = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=torch.tensor([[h, w]], device=device),
        input_ids=inputs["input_ids"],
        text_threshold=text_threshold,
    )[0]


    boxes = processed["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = processed["scores"].detach().cpu().numpy().astype(np.float32)
    keep = scores > box_threshold
    return boxes[keep], scores[keep]


# -------------------------------
# NMS helper
# -------------------------------
def nms_xyxy(boxes, scores, iou_thresh=0.5, topk=3):
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)
    keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou_thresh).numpy()
    return keep[:topk]


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--components_json", required=True)
    ap.add_argument("--out_dir", default="masks_out")

    ap.add_argument("--grounding_model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--sam_ckpt", required=True)

    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.20)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--max_boxes_per_label", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    image_pil = Image.open(args.image).convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_rgb.shape[:2]

    components = load_components(args.components_json)
    processor, dino = load_grounding_dino(args.grounding_model, device)
    predictor = load_sam(args.sam_type, args.sam_ckpt, device)
    predictor.set_image(image_rgb)

    results_manifest = {}
    per_component_masks = []

    for comp in components:
        boxes_xyxy, scores = dino_detect_boxes_for_phrase(
            image_pil, comp.lower(), processor, dino, device,
            args.box_threshold, args.text_threshold
        )

        if len(boxes_xyxy) == 0:
            print(f"[WARN] No detection for '{comp}'")
            per_component_masks.append(None)
            results_manifest[comp] = {"boxes": [], "scores": [], "mask_path": None}
            continue

        keep_idx = nms_xyxy(boxes_xyxy, scores, iou_thresh=args.nms_iou, topk=args.max_boxes_per_label)
        boxes_kept = boxes_xyxy[keep_idx]
        scores_kept = scores[keep_idx]

        boxes_torch = torch.from_numpy(boxes_kept).to(device)
        trans_boxes = predictor.transform.apply_boxes_torch(boxes_torch, (H, W))

        masks_best = []
        for b in trans_boxes:
            masks, ious, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=b[None, :],
                multimask_output=True
            )
            best_idx = int(torch.argmax(ious[0]).item())
            best_mask = masks[0, best_idx].detach().cpu().numpy().astype(bool)
            masks_best.append(best_mask)

        merged = np.zeros((H, W), dtype=bool)
        for m in masks_best:
            merged |= m

        comp_slug = "".join(c.lower() if c.isalnum() else "_" for c in comp).strip("_")
        mask_path = os.path.join(args.out_dir, f"{comp_slug}_mask.png")
        save_mask_png(merged, mask_path)

        results_manifest[comp] = {
            "boxes": boxes_kept.tolist(),
            "scores": scores_kept.tolist(),
            "mask_path": mask_path,
            "image_size": [H, W],
        }
        per_component_masks.append(merged)

    overlay_path = os.path.join(args.out_dir, "overlay.png")
    save_overlay(image_bgr, per_component_masks, components, overlay_path)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results_manifest, f, indent=2)

    print(f"Done. Masks in: {args.out_dir}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()



# python grounded_sam_sketch.py   --image 0.png   --components_json components.json   --grounding_model IDEA-Research/grounding-dino-base   --sam_ckpt ./sam_vit_h_4b8939.pth