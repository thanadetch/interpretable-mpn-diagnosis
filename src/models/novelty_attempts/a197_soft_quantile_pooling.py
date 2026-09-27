"""a197 - Soft-quantile pooling. NEW family: differentiable per-feature quantile (not mean/max).

For each feature dimension, aggregate the bag by a learnable QUANTILE of the patch values (soft, via a
sorting-free Gaussian-kernel CDF estimator) instead of mean/attention. q->0.5 = median (robust to
outlier patches like bone), q->1 = max (peak severity). One learnable q. Distinct from rank-softmax
(which reweights patches by rank) -- this reads a chosen order-statistic per feature. Concept-free,
self-contained, permutation/size-invariant, deterministic.
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
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.q_raw = nn.Parameter(torch.tensor(0.0))           # sigmoid -> q=0.5 init (median)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                          # [N,D]
        n = h.shape[0]
        q = torch.sigmoid(self.q_raw)                          # target quantile in (0,1)
        # soft-quantile per feature: weight patches by how close their within-feature rank is to q
        rank = (h.unsqueeze(0) < h.unsqueeze(1)).float().mean(dim=1)   # [N,D] approx empirical CDF/rank
        wq = torch.exp(-((rank - q) ** 2) / (2 * (0.15 ** 2)))         # [N,D] kernel around quantile q
        wq = wq / wq.sum(dim=0, keepdim=True).clamp(min=1e-8)
        z = (wq * h).sum(dim=0)                                # [D] soft-quantile per feature
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
