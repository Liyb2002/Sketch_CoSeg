# coseg/model.py
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=None, p=0.1):
        super().__init__()
        hidden = hidden or max(256, out_dim * 2)
        self.seq = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.out_norm(self.seq(x))

class CoSegModule(nn.Module):
    """
    Two-modality version (mask + box). Input Z is 1024-d = 512 + 512.
    Outputs binary probs A = [p_non, p_true] and projected descriptors Y.
    """
    def __init__(self, in_dim=1024, d=256, K=2, dropout=0.1, tau=0.7):
        super().__init__()
        assert in_dim == 1024, "Expecting 2x512 (mask+box). Adjust dataset or this check."

        self.d = d
        self.K = K
        self.tau = tau  # temperature to sharpen logits a bit

        # modality projectors
        self.proj_mask = MLP(512, d, p=dropout)
        self.proj_box  = MLP(512, d, p=dropout)

        # learnable gates over the two modalities (start equal)
        logits = torch.log(torch.tensor([0.5, 0.5]))  # inverse softmax init
        self.gates_logits = nn.Parameter(logits)

        # binary head -> [p_non, p_true]
        self.cls = nn.Linear(d, 2)

    @torch.no_grad()
    def _split(self, Z):
        return torch.split(Z, 512, dim=1)  # (z_mask, z_box)

    def forward(self, Z, img_ids):
        z_mask, z_box = self._split(Z)

        y_mask = self.proj_mask(z_mask)  # (N, d)
        y_box  = self.proj_box(z_box)    # (N, d)

        g = F.softmax(self.gates_logits, dim=0)  # (2,)
        Y = g[0]*y_mask + g[1]*y_box
        Y = F.normalize(Y, dim=1)

        logits = self.cls(Y) / self.tau
        A = F.softmax(logits, dim=-1)   # (N, 2): [p_non, p_true]
        return A, Y, logits

    @torch.no_grad()
    def gate_weights(self):
        g = F.softmax(self.gates_logits, dim=0)
        return {"w_mask": float(g[0]), "w_box": float(g[1])}
