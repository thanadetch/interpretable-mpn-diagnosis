"""a212 - 1.5-entmax attention pooling. NEW: between softmax (dense) and sparsemax (sparse).

entmax-1.5 (Peters et al. 2019) is the alpha=1.5 Tsallis-entropy attention: sparser than softmax but
smoother than sparsemax, giving a moderately sparse support that may suit "a few-but-not-one salient
regions". A distinct activation from both softmax (a-series) and sparsemax (a186). Implemented via the
exact alpha=1.5 thresholding (pure torch). Concept-free, self-contained, permutation/size-invariant, det.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax15(z: torch.Tensor) -> torch.Tensor:
    # exact 1.5-entmax over a 1-D vector (Peters et al. 2019)
    z = z / 2.0
    zs, _ = torch.sort(z, descending=True)
    rng = torch.arange(1, z.shape[0] + 1, device=z.device, dtype=z.dtype)
    mean = zs.cumsum(0) / rng
    mean_sq = (zs ** 2).cumsum(0) / rng
    ss = rng * (mean_sq - mean ** 2)
    delta = (1 - ss) / rng
    delta = delta.clamp(min=0)
    tau = mean - torch.sqrt(delta)
    support = (tau <= zs)
    k = support.sum().clamp(min=1)
    tau_star = tau[k.long() - 1]
    return torch.clamp(z - tau_star, min=0) ** 2


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax15(e)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
