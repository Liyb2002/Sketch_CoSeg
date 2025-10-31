#!/usr/bin/env python3
"""
visualize.py
Show masks one-by-one, blocking until you CLOSE the window.

Inputs:
  sketch/0.png
  sketch/out/segmentation.json  (from segment step)

Usage:
  python visualize.py
  # headless save only:
  python visualize.py --save_only
"""

import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMG_PATH = Path("sketch/0.png")
OUT_DIR  = Path("sketch/out")

def load_mask(path: Path) -> np.ndarray:
    """Load a binary mask (0/255) -> (H,W) uint8 in {0,1}."""
    m = Image.open(path).convert("L")
    arr = (np.array(m, dtype=np.uint8) > 127).astype(np.uint8)
    return arr

def resize_mask_to_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    """Resize mask to the image size using NEAREST to preserve binarity."""
    ih, iw = image.height, image.width
    mh, mw = mask.shape
    if (mh, mw) == (ih, iw):
        return mask
    pil_m = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    pil_m = pil_m.resize((iw, ih), Image.NEAREST)
    return (np.array(pil_m, dtype=np.uint8) > 127).astype(np.uint8)

def make_panel(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Overlay panel for a single part (mask-highlight)."""
    mask = resize_mask_to_image(mask, image)
    base = np.array(image).astype(np.uint8)
    vis  = (0.85 * base).astype(np.uint8)  # slight dim for contrast

    # tint in masked region
    tint = np.zeros_like(vis)
    tint[..., 0] = 220  # R
    tint[..., 1] = 120  # G
    tint[..., 2] =  80  # B
    mask3 = np.repeat(mask[..., None], 3, axis=2).astype(bool)
    vis[mask3] = (0.6 * vis[mask3] + 0.4 * tint[mask3]).astype(np.uint8)

    return Image.fromarray(vis)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=IMG_PATH)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--save_only", action="store_true",
                    help="save panels without showing (no blocking)")
    args = ap.parse_args()

    seg_path = args.out_dir / "segmentation.json"
    if not seg_path.exists():
        raise SystemExit(f"Missing {seg_path}. Run the segmenter first.")

    meta = json.loads(seg_path.read_text())
    labels = meta.get("labels", [])
    label2file = meta.get("masks", {})
    if not labels or not label2file:
        raise SystemExit("No labels/masks found in segmentation.json")

    image = Image.open(args.image).convert("RGB")
    save_dir = args.out_dir / "vis_parts"
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, lbl in enumerate(labels, 1):
        mpath = Path(label2file.get(lbl, ""))
        if not mpath.exists():
            print(f"[warn] mask missing for '{lbl}': {mpath}")
            continue

        mask  = load_mask(mpath)
        panel = make_panel(image, mask)

        print(f"[{i}/{len(labels)}] {lbl}")
        out_file = save_dir / f"{i:02d}_{lbl.replace(' ', '_')}.png"
        panel.save(out_file)

        if not args.save_only:
            plt.figure(figsize=(6, 6))
            plt.imshow(panel)
            plt.axis("off")
            plt.title(lbl)
            plt.tight_layout()
            # 👇 block until the figure window is CLOSED by you
            plt.show()  # blocking call; next mask only after you close this window
            plt.close()

    print(f"Saved per-part panels to: {save_dir}")

if __name__ == "__main__":
    main()
