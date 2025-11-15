# coseg/utils.py

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch


def _combine_and_norm_embeddings(emb_mask: np.ndarray, emb_box: np.ndarray) -> np.ndarray:
    """
    Combine mask + box embeddings and L2-normalize.
    Mirrors your current setup: (emb_mask + emb_box) / 2, then normalize.
    """
    emb_mask = emb_mask.astype(np.float32).ravel()
    emb_box = emb_box.astype(np.float32).ravel()
    emb = (emb_mask + emb_box) / 2.0
    norm = np.linalg.norm(emb)
    if norm < 1e-8:
        return emb
    return emb / norm


def load_coseg_dataset(outputs_dir: Path):
    """
    Load all fragment embeddings from:
        outputs/{x}/embeddings/{label_slug}/*.npz

    Returns:
        sketch_ids: List[str]
        frag_embs:  List[torch.FloatTensor]     # per sketch, (Mi, D)
        frag_labels: List[torch.LongTensor]     # per sketch, (Mi,)
        frag_info:  List[List[Dict[str, Any]]]  # per sketch, per fragment:
                                                #   { "x_id", "label_slug", "npz_path" }
        label_to_idx: Dict[str, int]            # label_slug -> idx in [0, n-1]
        emb_dim: int                            # D
    """
    outputs_dir = Path(outputs_dir)
    shape_to_embs: Dict[str, List[np.ndarray]] = {}
    shape_to_labels: Dict[str, List[int]] = {}
    shape_to_info: Dict[str, List[Dict[str, Any]]] = {}
    label_to_idx: Dict[str, int] = {}

    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    for x_dir in sorted(outputs_dir.iterdir()):
        if not x_dir.is_dir():
            continue

        x_id = x_dir.name
        emb_root = x_dir / "embeddings"
        if not emb_root.exists() or not emb_root.is_dir():
            continue

        per_shape_embs: List[np.ndarray] = []
        per_shape_labels: List[int] = []
        per_shape_info: List[Dict[str, Any]] = []

        # Each subdir in embeddings/ is a label slug
        for label_dir in sorted(emb_root.iterdir()):
            if not label_dir.is_dir():
                continue

            label_slug = label_dir.name
            if label_slug not in label_to_idx:
                label_to_idx[label_slug] = len(label_to_idx)
            lbl_idx = label_to_idx[label_slug]

            npz_paths = sorted(label_dir.glob("*.npz"))
            if not npz_paths:
                continue

            for p in npz_paths:
                data = np.load(p, allow_pickle=True)
                if "emb_mask" not in data or "emb_box" not in data:
                    continue

                emb_mask = data["emb_mask"]
                emb_box = data["emb_box"]
                emb = _combine_and_norm_embeddings(emb_mask, emb_box)

                per_shape_embs.append(emb)
                per_shape_labels.append(lbl_idx)
                per_shape_info.append({
                    "x_id": x_id,
                    "label_slug": label_slug,
                    "npz_path": str(p),
                })

        if per_shape_embs:
            shape_to_embs[x_id] = per_shape_embs
            shape_to_labels[x_id] = per_shape_labels
            shape_to_info[x_id] = per_shape_info

    if not shape_to_embs:
        raise RuntimeError(f"No embeddings found under {outputs_dir}/*/embeddings/*/")

    sketch_ids = sorted(shape_to_embs.keys())
    frag_embs: List[torch.Tensor] = []
    frag_labels: List[torch.Tensor] = []
    frag_info: List[List[Dict[str, Any]]] = []

    for x_id in sketch_ids:
        embs_np = np.stack(shape_to_embs[x_id], axis=0)  # (Mi, D)
        labels_np = np.array(shape_to_labels[x_id], dtype=np.int64)
        frag_embs.append(torch.from_numpy(embs_np).float())
        frag_labels.append(torch.from_numpy(labels_np).long())
        frag_info.append(shape_to_info[x_id])

    emb_dim = frag_embs[0].shape[1]
    return {
        "sketch_ids": sketch_ids,
        "frag_embs": frag_embs,
        "frag_labels": frag_labels,
        "frag_info": frag_info,
        "label_to_idx": label_to_idx,
        "emb_dim": emb_dim,
    }
