#!/usr/bin/env python

"""
Build 4×512-D CLIP descriptors for seat masks.

Folder structure:
  ./inputs/
      0/img.png
      0/mask.png
      1/img.png
      1/mask.png
      ...

For each folder i, we output a 2048-D descriptor:
  concat( component_embed, context_embed, global_embed, text_embed )

Using:
  - CLIP ViT-B/32 (512-D)
  - Hardcoded text prompt: "a motobike seat"
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import clip  # pip install git+https://github.com/openai/CLIP.git

# ---------------- CONFIG ----------------

INPUT_ROOT = "./inputs"
OUTPUT_NPY = "./seat_descriptors.npy"
OUTPUT_JSON = "./seat_descriptors_index.json"

OBJECT_NAME = "motobike"
LABEL_NAME = "seat"

MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONTEXT_SCALE = 1.5
NEUTRAL_GRAY = 127  # fill outside mask within tight crop

# -------------- MODEL LOADING ------------

print(f"Loading CLIP {MODEL_NAME} on {DEVICE}...")
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
model.eval()

@torch.no_grad()
def encode_image(pil_img):
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

@torch.no_grad()
def encode_text(text: str):
    tokens = clip.tokenize([text]).to(DEVICE)
    feat = model.encode_text(tokens)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu().numpy()

# ----------------- UTILS -----------------

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_mask(path: str) -> np.ndarray:
    # Convert to binary mask in {0,1}
    m = Image.open(path).convert("L")
    m = np.array(m)
    return (m > 128).astype(np.uint8)

def get_bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return x1, y1, x2, y2

def expand_bbox(x1, y1, x2, y2, scale, W, H):
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = int(max(0, cx - w / 2))
    nx2 = int(min(W, cx + w / 2))
    ny1 = int(max(0, cy - h / 2))
    ny2 = int(min(H, cy + h / 2))
    if nx2 <= nx1:
        nx2 = min(W, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(H, ny1 + 1)
    return nx1, ny1, nx2, ny2

def build_descriptor_for_sample(img: Image.Image,
                                mask: np.ndarray,
                                text_embed: np.ndarray,
                                global_embed: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    bbox = get_bbox_from_mask(mask)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox

    # ----- component embedding: tight masked crop -----
    img_np = np.array(img).copy()
    comp_crop_np = img_np[y1:y2, x1:x2].copy()
    comp_mask = mask[y1:y2, x1:x2]

    comp_crop_np[comp_mask == 0] = NEUTRAL_GRAY
    comp_crop = Image.fromarray(comp_crop_np)
    comp_embed = encode_image(comp_crop)  # (512,)

    # ----- context embedding: expanded bbox (unmasked) -----
    ex1, ey1, ex2, ey2 = expand_bbox(x1, y1, x2, y2, CONTEXT_SCALE, W, H)
    ctx_crop = img.crop((ex1, ey1, ex2, ey2))
    ctx_embed = encode_image(ctx_crop)  # (512,)

    # ----- final 2048-D descriptor -----
    desc = np.concatenate([comp_embed, ctx_embed, global_embed, text_embed], axis=0)
    # L2 normalize whole descriptor
    desc_norm = np.linalg.norm(desc) + 1e-8
    desc = desc / desc_norm
    return desc  # (2048,)

# ------------------ MAIN -----------------

def main():
    input_root = Path(INPUT_ROOT)
    text_prompt = f"a {OBJECT_NAME} {LABEL_NAME}"
    print(f"Using text prompt: '{text_prompt}'")

    text_embed = encode_text(text_prompt)

    descriptors = []
    index = []

    # sort folders numerically when possible
    def sort_key(name):
        return int(name) if name.isdigit() else name

    for name in sorted(os.listdir(input_root), key=sort_key):
        folder = input_root / name
        if not folder.is_dir():
            continue

        img_path = folder / "img.png"
        mask_path = folder / "mask.png"
        if not (img_path.exists() and mask_path.exists()):
            continue

        img = load_image(str(img_path))
        mask = load_mask(str(mask_path))

        # global embedding per image
        global_embed = encode_image(img)  # (512,)

        desc = build_descriptor_for_sample(
            img=img,
            mask=mask,
            text_embed=text_embed,
            global_embed=global_embed,
        )

        if desc is None:
            print(f"[skip] Empty mask in {folder}")
            continue

        descriptors.append(desc)
        index.append({
            "id": name,
            "img_path": str(img_path),
            "mask_path": str(mask_path),
        })

    if not descriptors:
        print("No valid descriptors generated.")
        return

    descriptors = np.stack(descriptors, axis=0)  # (N, 2048)
    np.save(OUTPUT_NPY, descriptors)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved descriptors: {descriptors.shape} -> {OUTPUT_NPY}")
    print(f"Wrote index for {len(index)} samples -> {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
