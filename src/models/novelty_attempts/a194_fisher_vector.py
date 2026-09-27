"""a194 - Fisher-vector-lite pooling. NEW family: 1st AND 2nd order residuals to learned Gaussians.

NetVLAD (a183) accumulated only FIRST-order residuals to cluster centres. a194 (Fisher-vector style)
adds the SECOND-order residual ((h-c)^2 - var) per soft-assigned Gaussian component, so the bag encodes
how patch features deviate from prototypal fibre patterns in BOTH mean and spread. Concept-free,
self-contained, permutation/size-invariant, deterministic.
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
        self.assign = nn.Linear(hidden_dim, K)
        self.centres = nn.Parameter(torch.randn(K, hidden_dim) * (hidden_dim ** -0.5))
        self.log_var = nn.Parameter(torch.zeros(K, hidden_dim))
        self.classifier = nn.Linear(2 * K * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        s = F.softmax(self.assign(h), dim=1)                # [N,K]
        var = self.log_var.exp().clamp(min=1e-4)            # [K,D]
        diff = h.unsqueeze(1) - self.centres.unsqueeze(0)   # [N,K,D]
        d1 = (s.unsqueeze(-1) * (diff / var.unsqueeze(0))).sum(0)            # [K,D] 1st order
        d2 = (s.unsqueeze(-1) * ((diff ** 2) / var.unsqueeze(0) - 1.0)).sum(0)  # [K,D] 2nd order
        fv = torch.cat([d1.reshape(-1), d2.reshape(-1)], dim=0)
        fv = torch.sign(fv) * torch.sqrt(fv.abs() + 1e-8)   # power norm
        fv = F.normalize(fv, dim=0)
        y = self.classifier(fv).view(-1)
        if return_attention:
            return y, s.sum(1), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
