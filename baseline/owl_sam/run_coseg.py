#!/usr/bin/env python
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

# ----------------- HARD-CODED CONFIG -----------------
INPUTS_DIR  = Path("inputs")
OUTPUTS_DIR = Path("outputs")
ENGINE_LABEL = "engine"   # label slug; embeddings live under embeddings/ENGINE_LABEL
KEEP_THRESHOLD = 0.5      # prob >= this => blue (keep), else red (reject)

# these are from your binary vis script
DIFF_THRESH = 12          # must match encoder.py
ALPHA_ACC   = 0.70        # opacity for "kept" (blue)
ALPHA_REJ   = 0.50        # opacity for "rejected" (red)

BLUE = (65, 105, 225)     # kept
RED  = (220, 20, 60)      # rejected
# -----------------------------------------------------


def load_embeddings_for_x(x_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all *.npz from:
        outputs/x/embeddings/ENGINE_LABEL/

    Each .npz is expected to contain: emb_mask, emb_box, meta.
    Returns a list of dicts: { "path", "emb", "meta" }.
    """
    items: List[Dict[str, Any]] = []

    emb_dir = x_dir / "embeddings" / ENGINE_LABEL
    if not emb_dir.exists() or not emb_dir.is_dir():
        return items

    npz_paths = sorted(emb_dir.glob("*.npz"))
    if not npz_paths:
        return items

    for p in npz_paths:
        data = np.load(p, allow_pickle=True)

        emb_mask = data["emb_mask"].astype(np.float32)
        emb_box = data["emb_box"].astype(np.float32)

        # Combine into one embedding vector per item
        emb = (emb_mask + emb_box) / 2.0
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        meta_raw = data["meta"]
        meta = json.loads(str(meta_raw))  # meta was saved via json.dumps

        items.append({
            "path": p,
            "emb": emb,
            "meta": meta,
        })

    return items


def compute_group_scores_0_to_1(embs: np.ndarray) -> np.ndarray:
    """
    Compute a 0-1 group score for each embedding:
        - 1.0 = most similar to group
        - 0.0 = least similar (outlier)

    Based on average cosine similarity.
    """
    N, D = embs.shape
    if N == 1:
        return np.array([1.0], dtype=np.float32)

    # cosine similarity matrix
    sim_matrix = embs @ embs.T

    # average similarity to others
    sim_sums = sim_matrix.sum(axis=1) - np.diag(sim_matrix)
    avg_sim = sim_sums / (N - 1)

    # min-max normalize to 0..1
    mn = avg_sim.min()
    mx = avg_sim.max()
    if mx - mn < 1e-12:
        return np.ones(N, dtype=np.float32)

    scores = (avg_sim - mn) / (mx - mn)
    return scores.astype(np.float32)


# ---------- helpers copied from your binary vis ----------

def _make_faint_for_recon(arr_rgb: np.ndarray) -> np.ndarray:
    """Exactly your _make_faint: used ONLY for mask reconstruction."""
    arr = arr_rgb.astype(np.float32)
    white_mask = (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
    fade = arr * 0.2 + 255.0 * 0.8
    arr[~white_mask] = fade[~white_mask]
    return np.clip(arr, 0, 255).astype(np.uint8)


def _reconstruct_fragment_mask_full(frag_overlay_path: Path,
                                    original_image_path: Path,
                                    diff_thresh: int = DIFF_THRESH) -> np.ndarray:
    overlay = np.array(Image.open(frag_overlay_path).convert("RGB"))
    orig = np.array(Image.open(original_image_path).convert("RGB"))
    faint = _make_faint_for_recon(orig)
    diff = np.abs(overlay.astype(np.int16) - faint.astype(np.int16))
    mask = (diff > diff_thresh).any(axis=2)
    return mask


def _alpha_blend(base: Image.Image, mask: np.ndarray, color, alpha: float) -> Image.Image:
    base = base.convert("RGBA")
    h, w = mask.shape
    overlay = Image.new("RGBA", (w, h), color + (0,))
    a = (mask.astype(np.uint8) * int(255 * alpha))
    overlay.putalpha(Image.fromarray(a, mode="L"))
    return Image.alpha_composite(base, overlay)


# ---------- our own fainting for the *final* sketch ----------

def _make_faint_for_vis(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Fade only non-white pixels to 50%, keep pure white background unchanged.
    """
    arr = arr_rgb.astype(np.float32)
    white_mask = (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
    arr[~white_mask] *= 0.5
    return np.clip(arr, 0, 255).astype(np.uint8)


# ---------- path helpers ----------

def _frag_overlay_path(x_dir: Path, meta: Dict[str, Any]) -> Path:
    """
    We assume overlay paths follow the pattern from your vis script:

        outputs/{x}/fragments/{label}/{label}_frag_{frag_idx}_overlay.png
    """
    label = meta.get("label_slug") or meta.get("label") or ENGINE_LABEL
    frag_idx = meta.get("frag_index", -1)
    return x_dir / "fragments" / str(label) / f"{label}_frag_{frag_idx}_overlay.png"


# ---------- visualization ----------

def visualize_x(x_dir: Path, x_id: str, items_sorted: List[Dict[str, Any]]) -> None:
    """
    For a given x_id:
      - load inputs/{x_id}.png
      - make non-white strokes 50% fainter, white background unchanged
      - reconstruct fragment mask for each item using its overlay.png
      - prob >= KEEP_THRESHOLD -> blue, else red
      - save to outputs/{x}/final_results/{ENGINE_LABEL}_overlay.png
    """
    sketch_path = INPUTS_DIR / f"{x_id}.png"
    if not sketch_path.exists():
        print(f"[WARN] Sketch not found for {x_id}: {sketch_path}")
        return

    # Load sketch and make it faint (for visualization)
    sketch = Image.open(sketch_path).convert("RGB")
    sketch_np = np.array(sketch)
    faint_np = _make_faint_for_vis(sketch_np)
    base = Image.fromarray(faint_np).convert("RGBA")

    out = base
    kept = rejected = 0

    for it in items_sorted:
        prob = it.get("prob", 0.0)

        overlay_path = _frag_overlay_path(x_dir, it["meta"])
        if not overlay_path.exists():
            # no overlay → cannot reconstruct mask
            print(f"[WARN] Missing overlay for x={x_id}: {overlay_path}")
            continue

        mask = _reconstruct_fragment_mask_full(overlay_path, sketch_path, diff_thresh=DIFF_THRESH)
        if mask.sum() == 0:
            continue

        if prob >= KEEP_THRESHOLD:
            out = _alpha_blend(out, mask, BLUE, ALPHA_ACC)
            kept += 1
        else:
            out = _alpha_blend(out, mask, RED, ALPHA_REJ)
            rejected += 1

    out_dir = x_dir / "final_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ENGINE_LABEL}_overlay.png"
    out.save(out_path)
    print(f"[vis] x={x_id} | kept={kept} rejected={rejected} → {out_path}")


# ---------- main loop ----------

def main():
    any_found = False

    # Loop over all x directories under outputs/
    for x_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not x_dir.is_dir():
            continue

        x_id = x_dir.name
        items = load_embeddings_for_x(x_dir)
        if not items:
            continue

        any_found = True
        embs = np.stack([it["emb"] for it in items], axis=0)  # (N, D)
        probs = compute_group_scores_0_to_1(embs)

        for it, p in zip(items, probs):
            it["prob"] = float(p)

        # Sort by prob descending (most in-group first)
        items_sorted = sorted(items, key=lambda x: -x["prob"])

        print("=====================================================")
        print(f"x_id = {x_id} | group folder = embeddings/{ENGINE_LABEL}")
        print("Higher prob = more likely to belong to this group")
        print("Lower prob = more likely to be the odd one out")
        print("=====================================================")

        def describe(it):
            meta = it["meta"]
            label = meta.get("label", meta.get("label_slug", ""))
            frag = meta.get("frag_index", "?")
            return f"label={label}, frag={frag}, file={it['path'].name}"

        for it in items_sorted:
            print(f"prob={it['prob']:.4f} :: {describe(it)}")

        print()  # blank line between x_ids

        # visualize using the same mask reconstruction pipeline as your binary script
        visualize_x(x_dir, x_id, items_sorted)

    if not any_found:
        print(f"[ERR] No embeddings found under {OUTPUTS_DIR}/x/embeddings/{ENGINE_LABEL}/")


if __name__ == "__main__":
    main()
