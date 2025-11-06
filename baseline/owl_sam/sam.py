#!/usr/bin/env python
# Minimal SAM wrapper; CLI overlays masks on both ctrl and sketch images with 50% background opacity.

import argparse
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, Optional

class SamRunner:
    def __init__(self, sam_type: str, sam_ckpt: str, device: torch.device, sketch_path: Optional[str] = None):
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        sam.to(device=device)
        self._pred = SamPredictor(sam)
        self.device = device

        self.sketch_bgr = None
        if sketch_path and Path(sketch_path).exists():
            sketch_rgb = np.array(Image.open(sketch_path).convert("RGB"))
            self.sketch_bgr = cv2.cvtColor(sketch_rgb, cv2.COLOR_RGB2BGR)

    def set_image(self, image_rgb: np.ndarray):
        self._pred.set_image(image_rgb)

    def mask_from_box(self, box_xyxy: np.ndarray, img_hw: Tuple[int, int]) -> np.ndarray:
        H, W = img_hw
        b = torch.from_numpy(box_xyxy[None, :].astype(np.float32)).to(self.device)
        bt = self._pred.transform.apply_boxes_torch(b, (H, W))
        masks, ious, _ = self._pred.predict_torch(
            point_coords=None, point_labels=None, boxes=bt, multimask_output=True
        )
        best = int(torch.argmax(ious[0]).item())
        return masks[0, best].detach().cpu().numpy().astype(bool)

    def overlay_mask(self, base_bgr: np.ndarray, mask: np.ndarray, color=(60, 180, 60)) -> np.ndarray:
        """Overlay mask with 50% faded background and 40% green tint."""
        faded = (base_bgr.astype(np.float32) * 0.5)  # 50% opacity background
        overlay = faded.copy()
        overlay[mask] = 0.6 * overlay[mask] + 0.4 * np.array(color, np.float32)
        return np.clip(overlay, 0, 255).astype(np.uint8)

# -------- CLI: overlays (single box) + keeps overlay.png --------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)  # ctrl image
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--box", nargs=4, type=float, required=True, help="x1 y1 x2 y2")
    ap.add_argument("--out_dir", default="./owl_sam_out")
    ap.add_argument("--sketch", default=None, help="path to original sketch image")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    img = Image.open(args.image).convert("RGB")
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = rgb.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ New way to initialize the runner (with sketch path)
    sam_runner = SamRunner(
        sam_type=args.sam_type,
        sam_ckpt=args.sam_ckpt,
        device=device,
        sketch_path=args.sketch,
    )
    sam_runner.set_image(rgb)

    box = np.array(args.box, np.float32)
    mask = sam_runner.mask_from_box(box, (H, W))

    # Overlay on ctrl image (with 50% faded background)
    overlay_ctrl = sam_runner.overlay_mask(bgr, mask)
    cv2.imwrite(str(Path(args.out_dir) / "overlay_ctrl.png"), overlay_ctrl)
    cv2.imwrite(str(Path(args.out_dir) / "overlay.png"), overlay_ctrl)

    # Overlay on sketch (if available)
    if sam_runner.sketch_bgr is not None:
        sketch_bgr = sam_runner.sketch_bgr
        if sketch_bgr.shape[:2] != (H, W):
            sketch_bgr = cv2.resize(sketch_bgr, (W, H), interpolation=cv2.INTER_LINEAR)
        overlay_sketch = sam_runner.overlay_mask(sketch_bgr, mask)
        cv2.imwrite(str(Path(args.out_dir) / "overlay_sketch.png"), overlay_sketch)

    print(f"[OK] overlays saved in {args.out_dir}")
