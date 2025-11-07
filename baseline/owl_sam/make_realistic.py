#!/usr/bin/env python
# Generate multiple photoreal variants from a sketch using SDXL+ControlNet.
# This module exposes one function: generate_variants(input_path, out_dir, style_prompts, seed).
# It does NOT hardcode any paths and does not print.

import os, math
from typing import List
import numpy as np
from PIL import Image
import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# Optional edge helpers
try:
    from controlnet_aux import HEDdetector
    _HED_OK = True
except Exception:
    _HED_OK = False

try:
    import cv2
    _CV2_OK = True
except Exception:
    _CV2_OK = False

# ----- model repo IDs (can stay hardcoded) -----
REALVIS_ID      = "SG161222/RealVisXL_V5.0"
SDXL_BASE_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROLNET_IDS = [
    "xinsir/controlnet-scribble-sdxl-1.0",
    "xinsir/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
]

# ----- default negative prompt & knobs (generic) -----
NEGATIVE = (
    "drawing, lineart, sketch, outline, cartoon, anime, cel shading, pencil, painting, illustration, "
    "grayscale, black and white, monochrome, lowres, low quality, low contrast, blurry, deformed, noisy, artifact, "
    "text, watermark, logo, caption, frame, border, label, numbers, symbols, signature, background clutter, "
    "abstract, surreal, unrealistic, extra limbs, cropped, partial, floating object, shadow mismatch"
)
STEPS       = 50
GUIDANCE    = 8.5
CTRL_SCALE  = 0.95
STRENGTH    = 0.4
MAX_SIDE    = 1024  # internal generation size; output will be resized back to sketch size

# ----- helpers -----
def _load_and_pad_white(path, max_side=MAX_SIDE):
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    scale = min(max_side / max(w0, h0), 1.0)
    if scale < 1.0:
        nw = max(64, int(math.floor(w0 * scale / 64.0) * 64))
        nh = max(64, int(math.floor(h0 * scale / 64.0) * 64))
        img = img.resize((nw, nh), Image.LANCZOS)
        w, h = img.size
    else:
        w, h = w0, h0
    pad_w = (64 - (w % 64)) % 64
    pad_h = (64 - (h % 64)) % 64
    if pad_w or pad_h:
        canvas = Image.new("RGB", (w + pad_w, h + pad_h), (255, 255, 255))
        canvas.paste(img, (0, 0))
        img = canvas
    return img, (w0, h0)

def _edges_hed(pil_img):
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_img)

def _edges_canny(pil_img, low=80, high=160):
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)  # dark edges on white
    return Image.fromarray(edges_rgb)

def _force_white_bg(pil_img, thresh=246):
    arr = np.array(pil_img).astype(np.uint8)
    mask = (arr > thresh).all(axis=2)
    arr[mask] = 255
    return Image.fromarray(arr)

def _load_controlnet(dtype):
    last = None
    for repo in CONTROLNET_IDS:
        try:
            return ControlNetModel.from_pretrained(repo, torch_dtype=dtype)
        except Exception as e:
            last = e
            continue
    raise last

def _build_pipes(device, dtype):
    # Try RealVis; else SDXL base
    try:
        controlnet = _load_controlnet(dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            REALVIS_ID, controlnet=controlnet, torch_dtype=dtype, add_watermarker=False
        ).to(device)
    except Exception:
        controlnet = _load_controlnet(dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_ID, controlnet=controlnet, torch_dtype=dtype, add_watermarker=False
        ).to(device)

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_REFINER_ID, torch_dtype=dtype, add_watermarker=False
    ).to(device)

    pipe.enable_attention_slicing(); refiner.enable_attention_slicing()
    for p in (pipe, refiner):
        try: p.enable_xformers_memory_efficient_attention()
        except Exception: pass
    return pipe, refiner

def generate_variants(
    input_path: str,
    out_dir: str,
    style_prompts: List[str],
    seed: int = 2025,
) -> List[str]:
    """
    Args:
      input_path: path to sketch (e.g., '0.png')
      out_dir: where to save ctrl_*.png
      style_prompts: list of positive prompts (each will be used once)
      seed: base RNG seed

    Returns:
      List of saved ctrl image paths, in order.
    """
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    init, orig_size = _load_and_pad_white(input_path, MAX_SIDE)

    # Edges (prefer HED; fallback Canny; fallback identity)
    if _HED_OK:
        try:
            edges = _edges_hed(init)
        except Exception:
            edges = _edges_canny(init) if _CV2_OK else init
    else:
        edges = _edges_canny(init) if _CV2_OK else init

    pipe, refiner = _build_pipes(device, dtype)

    saved = []
    for i, prompt in enumerate(style_prompts):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=init,
            control_image=edges,
            controlnet_conditioning_scale=CTRL_SCALE,
            strength=STRENGTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        ).images[0]

        refined = refiner(
            prompt=prompt, negative_prompt=NEGATIVE,
            image=out, strength=0.2,
            num_inference_steps=20, guidance_scale=5.0,
            generator=gen,
        ).images[0]

        final_img = _force_white_bg(refined).resize(orig_size, Image.LANCZOS)
        out_path = os.path.join(out_dir, f"ctrl_{i}.png")
        final_img.save(out_path)
        saved.append(out_path)

    # Free VRAM
    try:
        pipe.to("cpu"); refiner.to("cpu")
        del pipe, refiner
        torch.cuda.empty_cache()
    except Exception:
        pass

    return saved
