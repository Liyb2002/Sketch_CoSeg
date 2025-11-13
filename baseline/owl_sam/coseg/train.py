# coseg/train.py
import torch
from pathlib import Path
from .dataset import collect_embeddings
from .model import CoSegModule
from .losses import build_part_matrices, total_loss_rank_only   # <— change import
from .utils import save_assignments, log_vals

def run(
    outputs_dir: str = "outputs",
    label: str = "wheel",
    K: int = 2,                 # <— default binary: belongs vs not
    d: int = 256,
    epochs: int = 400,
    lr: float = 3e-4,
    device: str = "auto",
    save_dir: str = "outputs",
    dropout: float = 0.1,
    lambda_between: float = 0.2 # we keep separation term
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    Z, img_ids, frags, x_to_idx = collect_embeddings(Path(outputs_dir), label)
    net = CoSegModule(in_dim=1024, d=d, K=K, dropout=dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    Z = Z.to(device); img_ids = img_ids.to(device)

    for ep in range(1, epochs + 1):
        net.train(); opt.zero_grad()
        A, Y, _ = net(Z, img_ids)
        Ms = build_part_matrices(Y, A, img_ids, K=K)
        loss, logs = total_loss_rank_only(Ms, lambda_between=lambda_between)  # <— rank only
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()

        if ep % 20 == 0 or ep == 1 or ep == epochs:
            log_vals(ep, {"loss": loss, **logs})

    net.eval()
    with torch.no_grad():
        A, _, _ = net(Z, img_ids)
        save_assignments(frags, A.detach().cpu().numpy(), Path(save_dir), label, x_to_idx, K)
