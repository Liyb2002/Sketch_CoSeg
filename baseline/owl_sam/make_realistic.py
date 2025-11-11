#!/usr/bin/env python
# Generate multiple photoreal variants from a sketch using SDXL+ControlNet.
# Exposes: generate_variants(input_path, out_dir, style_prompts, seed)
# No hardcoded paths, no prints.

import os, math
from typing import List, Tuple
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

# ----- model repo IDs -----
REALVIS_ID      = "SG161222/RealVisXL_V5.0"
SDXL_BASE_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROLNET_IDS = [
    "xinsir/controlnet-scribble-sdxl-1.0",
    "xinsir/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
]

# ----- default negative prompt & knobs -----
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
MAX_SIDE    = 1024  # max internal size

# ----- helpers -----
def _load_and_pad_white(path: str, max_side: int = MAX_SIDE) -> Tuple[Image.Image, Tuple[int,int], Tuple[int,int]]:
    """
    Load sketch, optionally downscale uniformly to fit within max_side,
    then pad to multiples of 64 WITHOUT extra rescaling.

    Returns:
      padded_img  : PIL RGB
      orig_size   : (w0, h0)
      scaled_size : (ws, hs) before padding
    """
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size

    # Uniform scale if needed, preserve aspect
    scale = min(max_side / max(w0, h0), 1.0)
    if scale < 1.0:
        ws = int(round(w0 * scale))
        hs = int(round(h0 * scale))
        img = img.resize((ws, hs), Image.LANCZOS)
    else:
        ws, hs = w0, h0

    # Pad up to 64-multiple by adding white borders (no further rescale)
    pad_w = (64 - (ws % 64)) % 64
    pad_h = (64 - (hs % 64)) % 64
    Wp, Hp = ws + pad_w, hs + pad_h

    canvas = Image.new("RGB", (Wp, Hp), (255, 255, 255))
    canvas.paste(img, (0, 0))

    return canvas, (w0, h0), (ws, hs)

def _edges_hed(pil_img: Image.Image) -> Image.Image:
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_img)

def _edges_canny(pil_img: Image.Image, low=80, high=160) -> Image.Image:
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)  # dark edges on white → invert if needed
    return Image.fromarray(edges_rgb)

def _force_white_bg(pil_img: Image.Image, thresh: int = 246) -> Image.Image:
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
    raise last

def _build_pipes(device, dtype):
    # Try RealVis; else SDXL base
    try:
        controlnet = _load_controlnet(dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            REALVIS_ID,
            controlnet=controlnet,
            torch_dtype=dtype,
            add_watermarker=False,
        ).to(device)
    except Exception:
        controlnet = _load_controlnet(dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_BASE_ID,
            controlnet=controlnet,
            torch_dtype=dtype,
            add_watermarker=False,
        ).to(device)

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_REFINER_ID,
        torch_dtype=dtype,
        add_watermarker=False,
    ).to(device)

    pipe.enable_attention_slicing()
    refiner.enable_attention_slicing()
    for p in (pipe, refiner):
        try:
            p.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

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
      style_prompts: list of positive prompts (each used once)
      seed: base RNG seed

    Returns:
      List of saved ctrl image paths, in order.
    """
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    # Load + scale + pad
    init_padded, orig_size, scaled_size = _load_and_pad_white(input_path, MAX_SIDE)
    w0, h0 = orig_size
    ws, hs = scaled_size

    # Edges (prefer HED; fallback Canny; fallback identity)
    if _HED_OK:
        try:
            edges = _edges_hed(init_padded)
        except Exception:
            edges = _edges_canny(init_padded) if _CV2_OK else init_padded
    else:
        edges = _edges_canny(init_padded) if _CV2_OK else init_padded

    pipe, refiner = _build_pipes(device, dtype)

    saved = []
    for i, prompt in enumerate(style_prompts):
        gen = torch.Generator(device=device).manual_seed(seed + i)

        out = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=init_padded,
            control_image=edges,
            controlnet_conditioning_scale=CTRL_SCALE,
            strength=STRENGTH,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen,
        ).images[0]

        refined = refiner(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            image=out,
            strength=0.2,
            num_inference_steps=20,
            guidance_scale=5.0,
            generator=gen,
        ).images[0]

        # 1) remove padding: keep only the region that corresponds to scaled sketch
        refined_cropped = refined.crop((0, 0, ws, hs))

        # 2) if we downscaled originally, scale back to original sketch size
        if (ws, hs) != (w0, h0):
            refined_cropped = refined_cropped.resize((w0, h0), Image.LANCZOS)

        # 3) clean background
        final_img = _force_white_bg(refined_cropped)

        out_path = os.path.join(out_dir, f"ctrl_{i}.png")
        final_img.save(out_path)
        saved.append(out_path)

    # Free VRAM
    try:
        pipe.to("cpu"); refiner.to("cpu")
        del pipe, refiner
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return saved
