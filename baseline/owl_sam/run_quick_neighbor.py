#!/usr/bin/env python
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# ----------------- HARD-CODED CONFIG -----------------
OUTPUTS_DIR = Path("outputs")
ENGINE_LABEL = "engine"   # <- change this if your label slug is different
# -----------------------------------------------------


def load_embeddings_for_x(x_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all *.npz from:
        outputs/x/embeddings/ENGINE_LABEL/

    Each .npz is expected to contain: emb_mask, emb_box, emb_text, meta.
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

    if not any_found:
        print(f"[ERR] No embeddings found under {OUTPUTS_DIR}/x/embeddings/{ENGINE_LABEL}/")


if __name__ == "__main__":
    main()
