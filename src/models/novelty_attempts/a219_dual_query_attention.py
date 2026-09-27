"""a219 - Dual learned-query attention (concat). NEW: two complementary global queries, concatenated.

Two independent learned queries each pool the bag by their own softmax attention; the two pooled vectors
are concatenated for the head. Lets the model maintain two distinct global "what to look for" views
(e.g. dense-fibre vs sparse-fibre) without collapsing them. Distinct from a190 (single query) and a209
(per-head linear scorer over the gated rep). Concept-free, self-contained, permutation/size-inv, det.
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
        self.q = nn.Parameter(torch.randn(2, hidden_dim) * (hidden_dim ** -0.5))
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        scores = h @ self.q.t() / (h.shape[1] ** 0.5)        # [N,2]
        a = F.softmax(scores, dim=0)                          # [N,2]
        z = (a.t() @ h).reshape(-1)                           # [2D]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a.mean(1), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
