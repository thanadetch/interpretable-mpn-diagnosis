"""a183 — NetVLAD pooling. NEW family: residual aggregation to K learned cluster centres (not a mean).

Soft-assign each patch to K clusters, accumulate RESIDUALS (h_i - centre_k) per cluster, intra- and
L2-normalise, concatenate. This encodes the DISTRIBUTION of patches around prototypal fibre-pattern
centres rather than a single weighted mean — a fundamentally different (and classic) MIL/retrieval
aggregator never tried here. Concept-free, self-contained, permutation/size-invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.assign = nn.Linear(hidden_dim, K)                       # soft-assignment logits
        self.centres = nn.Parameter(torch.randn(K, hidden_dim) * (hidden_dim ** -0.5))
        self.classifier = nn.Linear(K * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                # [N,D]
        s = F.softmax(self.assign(h), dim=1)                         # [N,K] soft assignment
        # VLAD residuals: V[k] = Σ_i s[i,k] (h_i - c_k)
        # = (s^T h) - (Σ_i s[i,k]) c_k
        V = s.t() @ h - s.sum(dim=0, keepdim=True).t() * self.centres   # [K,D]
        V = F.normalize(V, dim=1)                                     # intra-normalise per cluster
        z = F.normalize(V.reshape(-1), dim=0)                         # global L2-normalise
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, s.sum(dim=1), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
