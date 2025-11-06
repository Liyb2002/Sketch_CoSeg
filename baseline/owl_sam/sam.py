#!/usr/bin/env python
# Minimal SAM wrapper; CLI now overlays masks on the image (no mask-only dump).

import argparse
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple

class SamRunner:
    def __init__(self, sam_type: str, sam_ckpt: str, device: torch.device):
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        sam.to(device=device)
        self._pred = SamPredictor(sam)
        self.device = device

    def set_image(self, image_rgb: np.ndarray):
        self._pred.set_image(image_rgb)

    def mask_from_box(self, box_xyxy: np.ndarray, img_hw: Tuple[int,int]) -> np.ndarray:
        H, W = img_hw
        b = torch.from_numpy(box_xyxy[None, :].astype(np.float32)).to(self.device)
        bt = self._pred.transform.apply_boxes_torch(b, (H, W))
        masks, ious, _ = self._pred.predict_torch(
            point_coords=None, point_labels=None, boxes=bt, multimask_output=True
        )
        best = int(torch.argmax(ious[0]).item())
        return masks[0, best].detach().cpu().numpy().astype(bool)

# -------- CLI: overlays (single box) + keeps overlay.png --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--box", nargs=4, type=float, required=True, help="x1 y1 x2 y2")
    ap.add_argument("--out_dir", default="./owl_sam_out")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = rgb.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = SamRunner(args.sam_type, args.sam_ckpt, device)
    runner.set_image(rgb)

    box = np.array(args.box, np.float32)
    mask = runner.mask_from_box(box, (H, W))

    # per-mask overlay
    overlay = bgr.copy().astype(np.float32)
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * np.array([60,180,60], np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    cv2.imwrite(str(Path(args.out_dir) / "sam_single_overlay.png"), overlay)
    cv2.imwrite(str(Path(args.out_dir) / "overlay.png"), overlay)  # keep overlay.png as well
    print(f"[OK] wrote {args.out_dir}/sam_single_overlay.png and overlay.png")
