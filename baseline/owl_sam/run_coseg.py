#!/usr/bin/env python3
# run_coseg.py — train co-seg MLP and save model only

from pathlib import Path

import torch

from coseg.model import CoSegMLP
from coseg.losses import coseg_cluster_loss
from coseg.utils import load_coseg_dataset

OUTPUTS_DIR = Path("outputs")

NUM_STEPS = 200
LR = 1e-3
HID_DIM = 128
OUT_DIM = 64

# you care more about intra-label compactness
LAMBDA_INTRA = 1.0
LAMBDA_INTER = 0.1
MARGIN = 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    dataset = load_coseg_dataset(OUTPUTS_DIR)
    sketch_ids = dataset["sketch_ids"]
    frag_embs = [e.to(device) for e in dataset["frag_embs"]]
    frag_labels = [l.to(device) for l in dataset["frag_labels"]]
    label_to_idx = dataset["label_to_idx"]
    emb_dim = dataset["emb_dim"]

    num_real_labels = len(label_to_idx)
    num_labels_total = num_real_labels + 1  # reserve last index for 'empty'
    empty_label_idx = num_labels_total - 1

    print(f"[INFO] Loaded {len(sketch_ids)} sketches")
    print(f"[INFO] Found {num_real_labels} real labels:")
    for lbl, idx in label_to_idx.items():
        print(f"    label '{lbl}' -> idx {idx}")
    print(f"[INFO] Total labels (including 'empty'): {num_labels_total}")
    print(f"[INFO] Reserved 'empty' label idx = {empty_label_idx} (no data yet).")

    model = CoSegMLP(in_dim=emb_dim, hid_dim=HID_DIM, out_dim=OUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---------- training ----------
    model.train()
    for step in range(1, NUM_STEPS + 1):
        optimizer.zero_grad()

        h_by_shape = [model(e) for e in frag_embs]

        total_loss, intra_loss, inter_loss = coseg_cluster_loss(
            h_by_shape=h_by_shape,
            labels_by_shape=frag_labels,
            num_labels=num_labels_total,
            lambda_intra=LAMBDA_INTRA,
            lambda_inter=LAMBDA_INTER,
            margin=MARGIN,
        )

        total_loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1 or step == NUM_STEPS:
            print(
                f"[step {step:04d}] "
                f"total={total_loss.item():.6f} "
                f"intra={intra_loss.item():.6f} "
                f"inter={inter_loss.item():.6f}"
            )

    # ---------- save model ----------
    coseg_dir = Path("coseg")
    coseg_dir.mkdir(parents=True, exist_ok=True)
    model_path = coseg_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "emb_dim": emb_dim,
            "hid_dim": HID_DIM,
            "out_dim": OUT_DIM,
            "label_to_idx": label_to_idx,
            "num_labels_total": num_labels_total,
        },
        model_path,
    )
    print(f"[DONE] Saved co-seg MLP to {model_path}")


if __name__ == "__main__":
    main()
