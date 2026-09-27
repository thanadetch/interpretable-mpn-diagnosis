"""a186 — Sparsemax attention pooling. NEW family: SPARSE (hard-subset) attention vs dense softmax.

All 197+ priors use softmax attention -> EVERY patch gets non-zero weight (dense). a186 replaces it
with sparsemax (Martins & Astudillo 2016): a Euclidean projection onto the simplex that yields EXACTLY
ZERO weight for low-scoring patches -> the pool is over a learned SPARSE SUBSET of patches, fully
differentiable and deterministic. Different selection geometry from any temperature/top-k variant
(top-k is fixed-size & non-differentiable; sparsemax learns the support size per bag). Concept-free,
self-contained, permutation/size-invariant, deterministic at inference.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(z: torch.Tensor) -> torch.Tensor:
    # z: [N] -> sparse weights on the simplex
    zs, _ = torch.sort(z, descending=True)
    cssv = zs.cumsum(0) - 1.0
    k = torch.arange(1, z.shape[0] + 1, device=z.device, dtype=z.dtype)
    support = (zs - cssv / k) > 0
    k_max = support.sum().clamp(min=1)
    tau = cssv[k_max.long() - 1] / k_max
    return torch.clamp(z - tau, min=0.0)


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
        a = sparsemax(e)                                 # sparse weights, sum to 1
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
