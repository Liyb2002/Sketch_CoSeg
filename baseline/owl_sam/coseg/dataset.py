# coseg/dataset.py
import os, json, glob
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

def _load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    # OLD (1536-d): z = np.concatenate([d["emb_mask"], d["emb_box"], d["emb_text"]], axis=0)
    # NEW (1024-d): drop text entirely
    z = np.concatenate([d["emb_mask"], d["emb_box"]], axis=0)  # (1024,)
    z = z.astype(np.float32)
    z /= (np.linalg.norm(z) + 1e-8)
    meta = json.loads(str(d["meta"].item()))
    return z, meta

def collect_embeddings(outputs_dir: Path, label_slug: str):
    """
    Returns:
      Z: (N, 1536) float32
      img_ids: (N,) long in [0..M-1] remapped
      frags: list of (x_id, frag_index, path)
      img_index_map: dict x_id -> 0..M-1
    """
    outputs_dir = Path(outputs_dir)
    # read reference set of x directories
    x_dirs = [p for p in sorted(outputs_dir.iterdir()) if p.is_dir()]
    feats, img_ids, frags = [], [], []
    x_to_idx: Dict[str, int] = {}
    next_idx = 0

    for xdir in x_dirs:
        x_id = xdir.name
        emb_root = xdir / "embeddings" / label_slug
        if not emb_root.exists():
            continue
        paths = sorted(emb_root.glob(f"{label_slug}_frag_*_embed.npz"))
        if not paths:
            continue
        if x_id not in x_to_idx:
            x_to_idx[x_id] = next_idx; next_idx += 1
        i_idx = x_to_idx[x_id]
        for p in paths:
            z, meta = _load_npz(p)
            feats.append(z)
            img_ids.append(i_idx)
            frags.append((x_id, int(meta.get("frag_index", -1)), str(p)))
    if not feats:
        raise SystemExit(f"No embeddings found for label '{label_slug}' under {outputs_dir}/**/embeddings/.")

    Z = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)
    img_ids = torch.tensor(np.array(img_ids, dtype=np.int64))
    return Z, img_ids, frags, x_to_idx
