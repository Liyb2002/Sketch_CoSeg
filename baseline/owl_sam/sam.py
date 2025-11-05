#!/usr/bin/env python
# Minimal SAM wrapper to get a single best mask per box, plus CLI.

import argparse
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

class SamRunner:
    def __init__(self, sam_type: str, sam_ckpt: str, device: torch.device):
        self.predictor = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        self.predictor.to(device=device)
        self._pred = SamPredictor(self.predictor)
        self.device = device
        self._size = None

    def set_image(self, image_rgb: np.ndarray):
        """
        image_rgb: HxWx3 uint8 RGB
        """
        self._pred.set_image(image_rgb)
        self._size = image_rgb.shape[:2]  # (H, W)

    def mask_from_box(self, box_xyxy: np.ndarray, img_hw: tuple[int,int]) -> np.ndarray:
        """
        Returns boolean HxW mask (best IoU among SAM's 3 proposals).
        """
        H, W = img_hw
        import torch
        b = torch.from_numpy(box_xyxy[None, :].astype(np.float32)).to(self.device)
        bt = self._pred.transform.apply_boxes_torch(b, (H, W))
        masks, ious, _ = self._pred.predict_torch(
            point_coords=None, point_labels=None, boxes=bt, multimask_output=True
        )
        best = int(torch.argmax(ious[0]).item())
        m = masks[0, best].detach().cpu().numpy().astype(bool)
        return m

# ---------- simple CLI test ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--sam_type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    # one box to test (x1 y1 x2 y2)
    ap.add_argument("--box", nargs=4, type=float, required=True)
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = SamRunner(args.sam_type, args.sam_ckpt, device)
    runner.set_image(img_np)

    box = np.array(args.box, np.float32)
    mask = runner.mask_from_box(box, (H, W))

    # save quick debug
    import cv2
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).astype(np.float32)
    bgr[mask] = 0.6 * bgr[mask] + 0.4 * np.array([60,180,60], np.float32)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    cv2.imwrite("sam_box_debug.png", bgr)
    print("[OK] wrote sam_box_debug.png")
