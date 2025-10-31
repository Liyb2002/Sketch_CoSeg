#!/usr/bin/env python3
import argparse
from pathlib import Path
from vlm_detect import detect_components
from segment import segment_with_text_prompts
from visualize import visualize_segmentation

def main():
    ap = argparse.ArgumentParser(description="Baseline-1: VLM->components, CLIPSeg->masks, visualize")
    ap.add_argument("--image", type=Path, default=Path("sketch/0.png"))
    ap.add_argument("--out_dir", type=Path, default=Path("sketch/out"))
    ap.add_argument("--max_parts", type=int, default=8)
    ap.add_argument("--use_local_clipseg", action="store_true",
                    help="use local transformers CLIPSeg fallback (if HF token missing)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) VLM -> components JSON
    comp_json = detect_components(args.image, out_dir=args.out_dir, max_parts=args.max_parts)

    # 2) Segmentation -> per-label masks
    masks_dir = segment_with_text_prompts(
        image_path=args.image,
        components_json=comp_json,
        out_dir=args.out_dir,
        use_local_fallback=args.use_local_clipseg
    )

    # 3) Visualization
    visualize_segmentation(
        image_path=args.image,
        masks_dir=masks_dir,
        out_dir=args.out_dir
    )

    print(f"done.\ncomponents: {comp_json}\nmasks dir: {masks_dir}\nvisuals in: {args.out_dir}")

if __name__ == "__main__":
    main()
