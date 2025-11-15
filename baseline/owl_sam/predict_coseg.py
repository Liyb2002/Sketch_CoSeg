#!/usr/bin/env python3
# predict_coseg.py — use trained co-seg MLP to assign fragments & visualize per label

from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from coseg.model import CoSegMLP
from coseg.utils import load_coseg_dataset

# ----------------- CONFIG -----------------
INPUTS_DIR  = Path("inputs")
OUTPUTS_DIR = Path("outputs")
MODEL_PATH  = Path("coseg/model.pt")

DIFF_THRESH = 12      # must match encoder / binary vis
ALPHA_LABEL = 0.7     # opacity for overlays
LABEL_COLOR = (65, 105, 225)  # blue-ish
# ------------------------------------------


# ---------- helpers copied/adapted from run_quick_neighbor ----------

def _make_faint_for_recon(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Exactly your _make_faint: used ONLY for mask reconstruction.
    """
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


def _make_faint_for_vis(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Fade only non-white pixels to 50%, keep pure white background unchanged.
    """
    arr = arr_rgb.astype(np.float32)
    white_mask = (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
    arr[~white_mask] *= 0.5
    return np.clip(arr, 0, 255).astype(np.uint8)


def _parse_frag_index_from_npz(npz_path: str) -> int:
    """
    Given something like .../{label}_frag_3_embed.npz, extract '3'.
    """
    stem = Path(npz_path).stem
    if "_frag_" not in stem:
        return -1
    try:
        return int(stem.split("_frag_")[1].split("_")[0])
    except Exception:
        return -1


def _overlay_path_from_info(x_id: str, info_j: dict) -> Path:
    """
    Reconstruct fragment overlay path from fragment info.
    We use the *original* label and npz filename to infer frag_index.

    overlays:
        outputs/{x}/fragments/{orig_label_slug}/{orig_label_slug}_frag_{k}_overlay.png
    """
    orig_label = info_j["label_slug"]
    npz_path = info_j["npz_path"]
    frag_idx = _parse_frag_index_from_npz(npz_path)
    return (
        OUTPUTS_DIR
        / x_id
        / "fragments"
        / orig_label
        / f"{orig_label}_frag_{frag_idx}_overlay.png"
    )


# ---------- main prediction + visualization ----------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Co-seg model not found at {MODEL_PATH}. Run run_coseg.py first.")

    # ---------- load dataset ----------
    dataset = load_coseg_dataset(OUTPUTS_DIR)
    sketch_ids = dataset["sketch_ids"]
    frag_embs = [e.to(device) for e in dataset["frag_embs"]]
    frag_labels = [l.to(device) for l in dataset["frag_labels"]]
    frag_info = dataset["frag_info"]
    label_to_idx = dataset["label_to_idx"]
    emb_dim = dataset["emb_dim"]

    num_real_labels = len(label_to_idx)
    idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}

    print(f"[INFO] Loaded {len(sketch_ids)} sketches for prediction")
    print(f"[INFO] Real labels:")
    for lbl, idx in label_to_idx.items():
        print(f"    label '{lbl}' -> idx {idx}")

    # ---------- load model ----------
    ckpt = torch.load(MODEL_PATH, map_location=device)
    hid_dim = ckpt.get("hid_dim", 128)
    out_dim = ckpt.get("out_dim", 64)

    if ckpt["emb_dim"] != emb_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: model was trained with emb_dim={ckpt['emb_dim']}, "
            f"but current embeddings have emb_dim={emb_dim}."
        )

    model = CoSegMLP(in_dim=emb_dim, hid_dim=hid_dim, out_dim=out_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---------- compute co-seg features ----------
    with torch.no_grad():
        h_by_shape = [model(e) for e in frag_embs]

    # ---------- compute centroids in co-seg space (per real label) ----------
    H_all = torch.cat(h_by_shape, dim=0)      # (N, F)
    y_all = torch.cat(frag_labels, dim=0)     # (N,)

    centroids = []
    for k in range(num_real_labels):
        mask_k = (y_all == k)
        if mask_k.any():
            c_k = H_all[mask_k].mean(dim=0)   # (F,)
            centroids.append(c_k)
        else:
            centroids.append(None)

    active_labels = [k for k in range(num_real_labels) if centroids[k] is not None]
    if not active_labels:
        print("[WARN] No active labels for centroid assignment; nothing to visualize.")
        return

    centroid_mat = torch.stack([centroids[k] for k in active_labels], dim=0)  # (L_active, F)

    # ---------- build per-sketch, per-assigned-label groups ----------
    # assigned_groups[x_id][label_slug] = list of fragment info dicts
    assigned_groups = {x_id: {} for x_id in sketch_ids}

    for x_id, h_i, labels_i, info_i in zip(sketch_ids, h_by_shape, frag_labels, frag_info):
        Mi = h_i.shape[0]

        # distances to active centroids: (Mi, L_active)
        diffs = h_i.unsqueeze(1) - centroid_mat.unsqueeze(0)   # (Mi, L_active, F)
        dists = torch.norm(diffs, dim=2)                       # (Mi, L_active)
        nearest_idx = dists.argmin(dim=1)                      # (Mi,)

        per_label_map = assigned_groups[x_id]

        for j in range(Mi):
            info_j = info_i[j]

            active_pos = int(nearest_idx[j].item())
            assigned_global_idx = active_labels[active_pos]
            assigned_slug = idx_to_label.get(assigned_global_idx, "unknown")

            per_label_map.setdefault(assigned_slug, []).append(info_j)

    # ---------- visualize per sketch, per assigned label ----------
    for x_id in sketch_ids:
        sketch_path = INPUTS_DIR / f"{x_id}.png"
        if not sketch_path.exists():
            print(f"[WARN] Sketch not found for {x_id}: {sketch_path}")
            continue

        sketch = Image.open(sketch_path).convert("RGB")
        sketch_np = np.array(sketch)
        faint_np = _make_faint_for_vis(sketch_np)

        x_dir = OUTPUTS_DIR / x_id
        out_dir = x_dir / "final_results"
        out_dir.mkdir(parents=True, exist_ok=True)

        label_map = assigned_groups[x_id]
        if not label_map:
            print(f"[INFO] No assigned fragments for x={x_id}, skipping visualization.")
            continue

        for label_slug, frag_list in label_map.items():
            if not frag_list:
                continue

            # start from faint sketch
            out_img = Image.fromarray(faint_np).convert("RGBA")

            kept = 0
            for info_j in frag_list:
                overlay_path = _overlay_path_from_info(x_id, info_j)
                if not overlay_path.exists():
                    print(f"[WARN] Missing overlay for x={x_id}, label={label_slug}: {overlay_path}")
                    continue

                mask = _reconstruct_fragment_mask_full(overlay_path, sketch_path, diff_thresh=DIFF_THRESH)
                if mask.sum() == 0:
                    print(f"[WARN] Empty mask in {overlay_path}")
                    continue

                out_img = _alpha_blend(out_img, mask, LABEL_COLOR, ALPHA_LABEL)
                kept += 1

            if kept == 0:
                continue

            out_path = out_dir / f"{label_slug}_coseg_overlay.png"
            out_img.save(out_path)
            print(f"[vis-coseg] x={x_id} label={label_slug} | fragments={kept} → {out_path}")


if __name__ == "__main__":
    main()
