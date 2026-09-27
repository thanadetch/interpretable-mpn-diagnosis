"""a233 - Learnable power-mean of a per-patch density vector. NEW: tune aggregation 'hardness' end-to-end.

Project each patch to a small non-negative density vector, then pool across patches with a generalized
(power) mean whose exponent p is LEARNABLE: p->1 = average density (diffuse/holistic, matches the grading
principle), p->large = soft-max (peak severity). The model picks where on the diffuse<->peak spectrum to
read fibre density. Distinct from a182 (power-mean of raw features): here it is over a learned density
representation. Concept-free, self-contained, permutation/size-invariant, deterministic, n=1 safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_dens=32):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.density = nn.Linear(hidden_dim, n_dens)
        self.log_p = nn.Parameter(torch.tensor(0.0))         # p = softplus(log_p)+1 in [1, inf)
        self.classifier = nn.Sequential(
            nn.Linear(n_dens, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_classes))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        d = torch.sigmoid(self.density(h))                   # [N,Dd] per-patch density in (0,1)
        p = F.softplus(self.log_p) + 1.0                     # learnable exponent >= 1
        pm = (d.clamp(min=1e-6).pow(p).mean(dim=0)).pow(1.0 / p)  # [Dd] power-mean over patches
        y = self.classifier(pm).view(-1)
        if return_attention:
            # contribution of each patch to the pooled density (for viz): mean over dims of d^p share
            w = d.pow(p).mean(dim=1)
            return y, w / w.sum().clamp(min=1e-8), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
