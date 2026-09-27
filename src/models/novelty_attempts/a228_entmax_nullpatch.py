"""a228 - entmax (a215) + null-patch abstain (a220). Combo of two GATE2-reachers (both passed virchow2 test)."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax15(z):
    z = z / 2.0
    zs, _ = torch.sort(z, descending=True)
    rng = torch.arange(1, z.shape[0] + 1, device=z.device, dtype=z.dtype)
    mean = zs.cumsum(0) / rng; mean_sq = (zs ** 2).cumsum(0) / rng
    ss = rng * (mean_sq - mean ** 2); delta = ((1 - ss) / rng).clamp(min=0)
    tau = mean - torch.sqrt(delta)
    k = (tau <= zs).sum().clamp(min=1)
    return torch.clamp(z - tau[k.long() - 1], min=0) ** 2


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.null_score = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        e_aug = torch.cat([e, self.null_score.view(1)], dim=0)
        a_aug = entmax15(e_aug); a = a_aug[:-1]
        z = torch.mv(h.t(), a) / a.sum().clamp(min=1e-6)
        y = self.classifier(z).view(-1)
        return (y, a, None) if return_attention else (y, None, None)


KWARGS = dict(input_dim=1280, num_classes=1)
