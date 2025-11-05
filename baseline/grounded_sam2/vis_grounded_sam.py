#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def load_results(out_dir: str):
    manifest_path = os.path.join(out_dir, "results.json")
    with open(manifest_path, "r") as f:
        return json.load(f)


def color_for(lbl: str) -> tuple[int, int, int]:
    # BGR ints for OpenCV, stable per label
    rng = np.random.default_rng(abs(hash(lbl)) % (2**32))
    c = rng.integers(50, 220, size=3, dtype=np.int32)
    return (int(c[0]), int(c[1]), int(c[2]))


def draw_outline(img_u8: np.ndarray, mask_bool: np.ndarray, color=(0, 255, 255), thickness=2):
    m_u8 = (mask_bool.astype(np.uint8) * 255)
    if m_u8.max() == 0:
        return
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(img_u8, cnts, -1, color, thickness)


def load_mask(mask_path: str, target_shape: tuple[int, int]) -> np.ndarray | None:
    if not mask_path or not os.path.exists(mask_path):
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != target_shape:
        m = cv2.resize(m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127)


def blend_mask(image_bgr_u8: np.ndarray, mask_bool: np.ndarray, color_bgr: tuple[int, int, int], alpha: float = 0.4):
    out = image_bgr_u8.astype(np.float32)
    col = np.array(color_bgr, np.float32)
    sel = mask_bool.astype(bool)
    if sel.any():
        out[sel] = (1.0 - alpha) * out[sel] + alpha * col
    return np.clip(out, 0, 255).astype(np.uint8)


def save_overlay_all(image_bgr: np.ndarray, labels: list[str], masks: list[np.ndarray | None], out_path: str):
    overlay = image_bgr.copy().astype(np.float32)

    # Blend each mask
    for m, lbl in zip(masks, labels):
        if m is None:
            continue
        col = np.array(color_for(lbl), np.float32)
        sel = m.astype(bool)
        if sel.any():
            overlay[sel] = 0.6 * overlay[sel] + 0.4 * col

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Legend
    y = 24
    for lbl in labels:
        col = color_for(lbl)
        cv2.rectangle(overlay, (10, int(y - 15)), (30, int(y + 5)), col, thickness=-1)
        cv2.putText(overlay, lbl, (40, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22

    cv2.imwrite(out_path, overlay)


def save_per_component_overlays(image_bgr: np.ndarray, labels: list[str], masks: list[np.ndarray | None], out_dir: str):
    paths = []
    tiles = []

    for m, lbl in zip(masks, labels):
        if m is None:
            continue
        col = color_for(lbl)
        vis = blend_mask(image_bgr, m, col, alpha=0.4)
        draw_outline(vis, m, color=(0, 255, 255), thickness=2)

        # Find a label placement (top-left of mask bbox)
        ys, xs = np.where(m)
        if len(xs) > 0:
            x0, y0 = int(xs.min()), int(ys.min())
            # label background
            text_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis, (x0, y0 - 18), (x0 + text_size[0] + 8, y0 - 2), (0, 0, 0), -1)
            cv2.putText(vis, lbl, (x0 + 4, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, lbl, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, lbl, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"{slugify(lbl)}_vis.png")
        cv2.imwrite(out_path, vis)
        paths.append(out_path)
        tiles.append(vis)

    # Collage
    if tiles:
        H, W = tiles[0].shape[:2]
        cols = min(4, max(1, int(math.ceil(math.sqrt(len(tiles))))))
        rows = int(math.ceil(len(tiles) / cols))
        canvas = np.ones((rows * H, cols * W, 3), dtype=np.uint8) * 255
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = tile
        cv2.imwrite(os.path.join(out_dir, "vis_collage.png"), canvas)

    return paths


def main():
    ap = argparse.ArgumentParser(description="Visualize Grounded-SAM masks without modifying generation code.")
    ap.add_argument("--image", required=True, help="Path to the original image (e.g., 0.png)")
    ap.add_argument("--out_dir", default="masks_out", help="Folder containing results.json and masks")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Load image
    image_rgb = np.array(Image.open(args.image).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    H, W = image_bgr.shape[:2]

    # Load results.json
    results = load_results(args.out_dir)
    labels = list(results.keys())

    # Load masks (ensure shape)
    masks = []
    for lbl in labels:
        info = results[lbl]
        mask_path = info.get("mask_path", None)
        m = load_mask(mask_path, (H, W))
        masks.append(m)

    # Save combined overlay (legend included)
    save_overlay_all(image_bgr, labels, masks, os.path.join(args.out_dir, "overlay.png"))

    # Save per-component + collage
    save_per_component_overlays(image_bgr, labels, masks, args.out_dir)

    print(f"Saved visualizations into: {args.out_dir}")


if __name__ == "__main__":
    main()

# python vis_grounded_sam.py --image 0.png --out_dir masks_out