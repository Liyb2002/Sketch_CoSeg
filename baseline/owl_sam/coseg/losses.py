# coseg/losses.py

import torch, torch.nn.functional as F

def second_singular_value(M):
    sv = torch.linalg.svdvals(M)
    return sv[1] if sv.numel() >= 2 else torch.tensor(0., device=M.device)

def build_part_matrices(Y, A, img_ids, K):
    device = Y.device
    M = int(img_ids.max().item() + 1)
    d = Y.size(1)
    Ms = []
    for k in range(K):
        rows = []
        for i in range(M):
            mask = (img_ids == i)
            Yi = Y[mask]
            Ai = A[mask, k:k+1]
            if Yi.numel() == 0:
                rows.append(torch.zeros(d, device=device))
                continue
            wsum = Ai.sum().clamp_min(1e-6)
            vk = (Ai.T @ Yi) / wsum
            rows.append(F.normalize(vk.squeeze(0), dim=0))
        Mk = torch.stack(rows, dim=0)
        Ms.append(Mk)
    return Ms

def rank_consistency_loss(Ms, lambda_between=0.2):
    # Lrank = sum_k σ2(Mk) - λ * sum_{k<l} σ2([Mk; Ml])
    K = len(Ms)
    loss_in = sum(second_singular_value(Mk) for Mk in Ms)
    loss_bt = 0.0
    for k in range(K):
        for l in range(k+1, K):
            loss_bt = loss_bt + second_singular_value(torch.cat([Ms[k], Ms[l]], dim=0))
    return loss_in - lambda_between * loss_bt

# ---- NEW: rank-only total ----
def total_loss_rank_only(Ms, lambda_between=0.2):
    Lrank = rank_consistency_loss(Ms, lambda_between=lambda_between)
    return Lrank, {"Lrank": Lrank.detach()}
