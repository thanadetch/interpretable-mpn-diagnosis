"""a190 - Modern-Hopfield pooling. NEW family: associative retrieval to a learned stored query.

A single learned query state retrieves a bag representation by (learnable-beta) softmax association
over the patches, optionally iterated. Equivalent to a 1-slot modern Hopfield network / learned-query
cross-attention -- distinct from per-patch gated attention (the query is GLOBAL & learned, not derived
per patch). Concept-free, self-contained, permutation/size-invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_iter=2):
        super().__init__()
        self.n_iter = n_iter
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.query = nn.Parameter(torch.randn(hidden_dim) * (hidden_dim ** -0.5))
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        beta = self.log_beta.exp().clamp(min=1e-2, max=50.0)
        q = self.query
        a = None
        for _ in range(self.n_iter):
            a = F.softmax(beta * (h @ q) / (h.shape[1] ** 0.5), dim=0)   # [N]
            q = torch.mv(h.t(), a)                          # retrieved state -> new query
        y = self.classifier(q).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
