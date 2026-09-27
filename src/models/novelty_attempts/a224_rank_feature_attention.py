"""a224 - Rank-transformed-feature attention. NEW: attend on per-feature within-bag RANKS (scale-free).

Replace each feature value by its within-bag rank (in [0,1]) before scoring/pooling, so the aggregator
reads relative ordering across patches rather than raw magnitudes (robust to per-bag scale/offset and to
outlier patches). Distinct from length-norm / rank-softmax (which rank PATCHES by a scalar score).
Concept-free, self-contained, permutation/size-invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        h = self.bottleneck(features)                        # [N,D]
        n = h.shape[0]
        # within-bag rank per feature in [0,1] (ties -> mean via < count); differentiable-friendly via sigmoid surrogate not needed for fwd
        rank = (h.unsqueeze(0) < h.unsqueeze(1)).float().mean(dim=1)   # [N,D] empirical CDF
        hr = h + (rank - 0.5)                                 # inject rank as additive scale-free signal
        a = F.softmax(self.attention_W(self.attention_V(hr) * self.attention_U(hr)).squeeze(-1), dim=0)
        z = torch.mv(hr.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
