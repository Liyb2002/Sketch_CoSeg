#!/usr/bin/env python
# 0.png -> ctrl.png using SDXL + ControlNet (scribble fallback to canny), no prints.

import os, math
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

# ----- hardcoded IO / models -----
INPUT_PATH  = "./0.png"
OUTPUT_PATH = "./ctrl.png"          # saved resized back to the original sketch size
EDGE_DEBUG  = "./owl_sam_out/edge_debug.png"

REALVIS_ID      = "SG161222/RealVisXL_V5.0"
SDXL_BASE_ID    = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID = "stabilityai/stable-diffusion-xl-refiner-1.0"

CONTROLNET_IDS = [
    "xinsir/controlnet-scribble-sdxl-1.0",
    "xinsir/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
]

PROMPT = (
    "photorealistic motorcycle product photo that matches the sketch silhouette and pose; "
    "realistic materials (painted metal tank, rubber tires, chrome details), soft studio lighting, "
    "pure white seamless background, 85mm lens"
)
NEGATIVE = (
    "drawing, lineart, sketch, outline, cartoon, anime, cel shading, pencil, grayscale, "
    "watermark, text, logo, frame, border, lowres, blurry, noisy background"
)

SEED        = 2025
STEPS       = 50
GUIDANCE    = 4.5
CTRL_SCALE  = 0.95
STRENGTH    = 0.92
MAX_SIDE    = 1024  # internal generation size; output will be resized back to the sketch size

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
    edges_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)
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

    pipe.enable_attention_slicing()
    refiner.enable_attention_slicing()
    for p in (pipe, refiner):
        try:
            p.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe, refiner

# ----- public API -----
def run():
    if not os.path.exists("./owl_sam_out"):
        os.makedirs("./owl_sam_out", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    gen    = torch.Generator(device=device).manual_seed(SEED)

    init, orig_size = _load_and_pad_white(INPUT_PATH, MAX_SIDE)

    if _HED_OK:
        try:
            edges = _edges_hed(init)
        except Exception:
            edges = _edges_canny(init) if _CV2_OK else init
    else:
        edges = _edges_canny(init) if _CV2_OK else init
    edges.save(EDGE_DEBUG)

    pipe, refiner = _build_pipes(device, dtype)

    out = pipe(
        prompt=PROMPT,
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
        prompt=PROMPT, negative_prompt=NEGATIVE,
        image=out, strength=0.25,
        num_inference_steps=20, guidance_scale=5.0,
        generator=gen,
    ).images[0]

    final_img = _force_white_bg(refined).resize(orig_size, Image.LANCZOS)
    final_img.save(OUTPUT_PATH)

def main():
    run()

if __name__ == "__main__":
    main()
