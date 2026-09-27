"""a185 — LogSumExp (smooth-max) pooling. NEW family: smooth maximum over patches, learnable sharpness.

z_j = (1/tau) * [ logsumexp_i(tau * h_ij) - log N ]   per feature j, with learnable tau>0.
tau->0 recovers the mean; tau->inf recovers the max. Unlike attention (which picks WHICH patches) or
power-mean (which needs h>=0), LSE is a smooth, numerically-stable soft-max over patches per feature
that can emphasise the single most-expressed patch per feature dimension. log N keeps it bag-size
invariant. Concept-free, self-contained, permutation/size-invariant, deterministic at inference.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.log_tau = nn.Parameter(torch.tensor(0.0))   # tau = exp(0) = 1 init

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        n = h.shape[0]
        tau = self.log_tau.exp().clamp(min=1e-2, max=50.0)
        z = (torch.logsumexp(tau * h, dim=0) - math.log(n)) / tau    # smooth-max per feature, size-invariant
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
