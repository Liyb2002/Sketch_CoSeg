# coseg/model.py

import torch
import torch.nn as nn


class CoSegMLP(nn.Module):
    """
    Tiny MLP that maps raw fragment embeddings to a co-segmentation space.

    in_dim  : original embedding dimension (from emb_mask/emb_box)
    hid_dim : hidden layer size
    out_dim : co-seg space dimension
    """

    def __init__(self, in_dim: int, hid_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_dim)
        returns: (..., out_dim)
        """
        return self.net(x)
