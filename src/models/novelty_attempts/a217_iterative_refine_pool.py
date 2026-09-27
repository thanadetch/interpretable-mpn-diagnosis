"""a217 - Iterative attention refinement. NEW: re-pool using the previous pooled vector as the query.

Start from a mean pool; for T steps, score each patch by similarity to the CURRENT bag vector, re-pool,
update. A fixed-point/deep-equilibrium-lite read where the saliency is defined RELATIVE to the emerging
bag representation (not a static learned scorer). Concept-free, self-contained, permutation/size-
invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_iter=3):
        super().__init__()
        self.n_iter = n_iter
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.log_s = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        k = self.proj(h)
        q = h.mean(0)                                       # init = mean pool
        s = self.log_s.exp().clamp(max=50.0)
        a = None
        for _ in range(self.n_iter):
            a = F.softmax(s * (k @ F.normalize(q, dim=0)) / (h.shape[1] ** 0.5), dim=0)
            q = torch.mv(h.t(), a)
        y = self.classifier(q).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
