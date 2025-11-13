#!/usr/bin/env python3
# run_quick_neighbor.py — zero-training classifier using only emb_box (local neighborhood).
# Builds a robust prototype per label and scores fragments by cosine similarity.

import os, json
import numpy as np
from pathlib import Path

OUTPUTS_DIR = Path("outputs")
LABEL       = "engine"     # <-- change your label
TRIM_KEEP   = 0.60         # keep top 60% by similarity each refinement
ITERS       = 3            # refinement iterations
SIG_SCALE   = 3.0
SIG_SHIFT   = 0.0

def _load_emb_box(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    # pick which embedding you want as your "feature"
    # z = d["emb_mask"].astype(np.float32)   # (512,) instead of emb_box
    z = d["emb_box"].astype(np.float32) # or text, if you prefer
    # z = d["emb_text"].astype(np.float32) # or text, if you prefer
    
    # z = np.random.randn(*z.shape).astype(np.float32)  # <-- sanity-test line, keep as-is for now

    n = np.linalg.norm(z) + 1e-8
    return z / n, json.loads(str(d["meta"].item()))

def _robust_centroid(Z, keep=0.6, iters=3):
    # Z: (N, 512) normalized
    c = Z.mean(axis=0)
    c /= (np.linalg.norm(c) + 1e-8)
    for _ in range(iters):
        sims = Z @ c  # cosine sim since normalized
        idx = np.argsort(sims)[::-1]
        k = max(1, int(len(idx) * keep))
        Zk = Z[idx[:k]]
        c = Zk.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-8)
    return c

def _sigmoid(x):  # stable-ish
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))

def main():
    # 1) Collect all emb_box for this LABEL
    records = []  # [(x_id, frag_idx, npz_path, z_box)]
    for x_dir in sorted([p for p in OUTPUTS_DIR.iterdir() if p.is_dir()]):
        emb_dir = x_dir / "embeddings" / LABEL
        if not emb_dir.exists():
            continue
        for p in sorted(emb_dir.glob(f"{LABEL}_frag_*_embed.npz")):
            z, meta = _load_emb_box(p)
            x_id = meta.get("x_id", x_dir.name)
            try:
                frag_idx = int(meta.get("frag_index", -1))
            except Exception:
                frag_idx = -1
            records.append((x_id, frag_idx, p, z))

    if not records:
        print(f"[quick] No embeddings found for label '{LABEL}'.")
        return

    Z = np.stack([r[3] for r in records], axis=0)  # (N,512)

    # 2) Build robust centroid
    proto = _robust_centroid(Z, keep=TRIM_KEEP, iters=ITERS)

    # 3) Score each fragment and produce p_true via sigmoid
    sims = Z @ proto  # cosine similarity in [-1,1]

    # Compute distribution stats of similarities
    mu = float(sims.mean())
    sigma = float(sims.std())

    # If everyone looks identical (no spread), treat as "perfect neighbors"
    if sigma < 1e-6:
        p_true = np.ones_like(sims, dtype=np.float32)
    else:
        # For general case: z-score and require large positive deviation
        zscore = (sims - mu) / (sigma + 1e-8)

        # How many std above mean counts as being a real neighbor
        K_SHIFT = 3.0  # increase to 4.0+ to be stricter

        logits = SIG_SCALE * (zscore - K_SHIFT)
        p_true = _sigmoid(logits).astype(np.float32)  # in (0,1)

    # 4) Save assignments in the same format used by the trainer
    for (x_id, frag_idx, npz_path, _), p in zip(records, p_true):
        co_dir = npz_path.parent.parent.parent / "co_seg"  # outputs/x/co_seg
        co_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "label_slug": LABEL,
            "x_id": str(x_id),
            "frag_index": int(frag_idx),
            "K": 2,
            "probs": [float(1.0 - p), float(p)],  # [p_non, p_true]
            "method": "quick_neighbor_cosine",
            "sim": float(sims[np.where(p_true==p)[0][0]]) if np.ndim(p_true)>0 else float(sims),
            "proto_trim_keep": TRIM_KEEP,
            "proto_iters": ITERS,
            "sig_scale": SIG_SCALE,
            "sig_shift": SIG_SHIFT,
        }
        out_path = co_dir / f"{LABEL}_frag_{frag_idx}_assign.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"[quick] Done. Wrote assignments under outputs/*/co_seg/ using emb_box-only.")

if __name__ == "__main__":
    main()
