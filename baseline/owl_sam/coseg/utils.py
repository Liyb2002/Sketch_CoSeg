# coseg/utils.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

def save_assignments(frags, part_probs, save_dir: Path, label_slug: str, x_to_idx: Dict[str,int], K: int):
    """
    frags: list of (x_id, frag_index, path_to_npz)
    part_probs: (N, K) numpy array
    """
    save_dir = Path(save_dir)
    for (x_id, frag_idx, npz_path), row in zip(frags, part_probs):
        out_dir = save_dir / x_id / "co_seg"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "label_slug": label_slug,
            "x_id": x_id,
            "frag_index": int(frag_idx),
            "K": K,
            "probs": row.tolist(),
        }
        with open(out_dir / f"{label_slug}_frag_{frag_idx}_assign.json", "w") as f:
            json.dump(out, f, indent=2)

def log_vals(step, scalars: dict):
    msg = f"[step {step}] " + " | ".join([f"{k}={float(v):.4f}" for k,v in scalars.items()])
    print(msg)
