# coseg/losses.py

from typing import List, Tuple

import torch


def coseg_cluster_loss(
    h_by_shape: List[torch.Tensor],
    labels_by_shape: List[torch.Tensor],
    num_labels: int,
    lambda_intra: float = 1.0,
    lambda_inter: float = 0.1,
    margin: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Clustering-style co-seg loss in the learned feature space:

        1) Intra-label compactness:
           - fragments in the same label should be close to their label centroid.

        2) Inter-label separation:
           - label centroids should be at least `margin` apart (hinge).

    Args:
        h_by_shape: list of (Mi, F) tensors, co-seg features per sketch.
        labels_by_shape: list of (Mi,) int tensors with values in [0, num_labels-1].
                         This includes the 'empty' label as one of the indices (even if unused).
        num_labels: total number of labels, including the empty label (n+1).
        lambda_intra: weight for the intra-label term (usually larger).
        lambda_inter: weight for the inter-label term (usually smaller).
        margin: desired minimum distance between label centroids.

    Returns:
        total_loss, intra_loss, inter_loss
    """
    device = h_by_shape[0].device

    # Flatten across all shapes: H_all: (N, F), y_all: (N,)
    H_all = torch.cat(h_by_shape, dim=0)          # (N, F)
    y_all = torch.cat(labels_by_shape, dim=0)     # (N,)

    # ----- compute centroids -----
    centroids = []
    for k in range(num_labels):
        mask_k = (y_all == k)
        if mask_k.any():
            c_k = H_all[mask_k].mean(dim=0)       # (F,)
            centroids.append(c_k)
        else:
            centroids.append(None)

    # ----- intra-label compactness -----
    intra_loss = torch.zeros((), device=device)
    intra_count = 0

    for k in range(num_labels):
        if centroids[k] is None:
            continue
        mask_k = (y_all == k)
        if mask_k.sum() == 0:
            continue
        diffs = H_all[mask_k] - centroids[k]      # (Nk, F)
        # mean squared distance within this label
        label_loss = (diffs.pow(2).sum(dim=1)).mean()
        intra_loss = intra_loss + label_loss
        intra_count += 1

    if intra_count > 0:
        intra_loss = intra_loss / intra_count

    # ----- inter-label separation (centroid margin loss) -----
    inter_loss = torch.zeros((), device=device)
    inter_count = 0

    # collect indices of labels that actually appear
    active_labels = [k for k in range(num_labels) if centroids[k] is not None]

    for i_idx, k in enumerate(active_labels):
        for l in active_labels[i_idx + 1 :]:
            ck = centroids[k]
            cl = centroids[l]
            if ck is None or cl is None:
                continue
            dist = torch.norm(ck - cl, p=2)
            # hinge: penalize if too close
            penalty = torch.relu(margin - dist).pow(2)
            inter_loss = inter_loss + penalty
            inter_count += 1

    if inter_count > 0:
        inter_loss = inter_loss / inter_count

    total_loss = lambda_intra * intra_loss + lambda_inter * inter_loss
    return total_loss, intra_loss, inter_loss
