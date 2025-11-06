# make_ctrl_pro.py
# One-click: reads 0.png -> writes ctrl.png (photoreal, white bg) using SDXL + ControlNet (scribble, fallback canny).
import os, math
import numpy as np
from PIL import Image
import torch

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
)

# --- optional deps; we fall back gracefully ---
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

# ---------------- Hardcoded IO/Models ----------------
INPUT_PATH  = "0.png"
OUTPUT_PATH = "ctrl.png"
EDGE_DEBUG  = "edge_debug.png"

# Photoreal SDXL base (newer, very realistic). If load fails, we fall back to Stability's SDXL base.
REALVIS_ID        = "SG161222/RealVisXL_V5.0"
SDXL_BASE_ID      = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER_ID   = "stabilityai/stable-diffusion-xl-refiner-1.0"

# Public SDXL ControlNets to try in order (scribble → canny):
CONTROLNET_IDS = [
    "xinsir/controlnet-scribble-sdxl-1.0",
    "xinsir/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
]

# Prompting tuned for white-back product realism:
PROMPT = (
    "photorealistic motorcycle product photo that matches the sketch silhouette and pose; "
    "realistic materials (painted metal tank, rubber tires, chrome details), soft studio lighting, "
    "subtle shadows, PURE WHITE seamless background, 85mm lens"
)
NEGATIVE = (
    "drawing, lineart, sketch, outline, cartoon, anime, cel shading, pencil, grayscale, "
    "watermark, text, logo, frame, border, lowres, blurry, deformed, noisy background"
)

# Generation knobs (tuned to escape 'sketch look'):
SEED        = 2025
STEPS       = 50
GUIDANCE    = 4.5
CTRL_SCALE  = 0.95   # <1.0 keeps silhouette but lets texture/lighting form
STRENGTH    = 0.92   # push away from input pixels -> more photoreal
MAX_SIDE    = 1024   # SDXL friendly (padded to multiples of 64)

# ---------------- Helpers ----------------
def _load_and_pad_white(path, max_side=MAX_SIDE):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    if scale < 1.0:
        nw = max(64, int(math.floor(w * scale / 64.0) * 64))
        nh = max(64, int(math.floor(h * scale / 64.0) * 64))
        img = img.resize((nw, nh), Image.LANCZOS)
        w, h = img.size
    pad_w = (64 - (w % 64)) % 64
    pad_h = (64 - (h % 64)) % 64
    if pad_w or pad_h:
        canvas = Image.new("RGB", (w + pad_w, h + pad_h), (255, 255, 255))
        canvas.paste(img, (0, 0))
        img = canvas
    return img

def _edges_hed(pil_img):
    if not _HED_OK:
        raise RuntimeError("HED not available")
    hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return hed(pil_img)  # PIL RGB, white bg + soft dark edges

def _edges_canny(pil_img, low=80, high=160):
    if not _CV2_OK:
        raise RuntimeError("OpenCV not available")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)  # dark edges on white bg
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
            print(f"[Info] Loading ControlNet: {repo}")
            return ControlNetModel.from_pretrained(repo, torch_dtype=dtype)
        except Exception as e:
            print(f"[Warn] ControlNet {repo} failed: {e}")
            last = e
    raise last

def _build_pipes(device, dtype):
    # Try photoreal RealVis XL first, then fallback to SDXL base if unavailable
    base_id = REALVIS_ID
    pipe = None
    try:
        controlnet = _load_controlnet(dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_id, controlnet=controlnet, torch_dtype=dtype, add_watermarker=False
        ).to(device)
        print(f"[Info] Using base: {base_id}")
    except Exception as e:
        print(f"[Warn] Failed to load {base_id}: {e}")
        base_id = SDXL_BASE_ID
        controlnet = _load_controlnet(dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_id, controlnet=controlnet, torch_dtype=dtype, add_watermarker=False
        ).to(device)
        print(f"[Info] Using base: {base_id}")

    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_REFINER_ID, torch_dtype=dtype, add_watermarker=False
    ).to(device)

    pipe.enable_attention_slicing(); refiner.enable_attention_slicing()
    for p in (pipe, refiner):
        try: p.enable_xformers_memory_efficient_attention()
        except Exception: pass
    return pipe, refiner

# ---------------- Main ----------------
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError("0.png not found")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    gen    = torch.Generator(device=device).manual_seed(SEED)

    # 1) Prep input + edges (prefer HED → smoother, less 'inky'; fallback to Canny)
    init  = _load_and_pad_white(INPUT_PATH, MAX_SIDE)
    try:
        edges = _edges_hed(init);  print("[Info] Using HED edges")
    except Exception:
        edges = _edges_canny(init); print("[Info] Using Canny edges")
    edges.save(EDGE_DEBUG)

    # 2) Build SDXL+ControlNet + refiner
    pipe, refiner = _build_pipes(device, dtype)

    # 3) Main generation (looser control, strong denoise to escape sketch look)
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

    # 4) Refiner polish (adds realistic materials/lighting detail)
    refined = refiner(
        prompt=PROMPT, negative_prompt=NEGATIVE,
        image=out, strength=0.25,
        num_inference_steps=20, guidance_scale=5.0,
        generator=gen,
    ).images[0]

    final_img = _force_white_bg(refined)
    final_img.save(OUTPUT_PATH)
    print(f"[Done] Saved: {OUTPUT_PATH}  (control preview: {EDGE_DEBUG})")

if __name__ == "__main__":
    main()
