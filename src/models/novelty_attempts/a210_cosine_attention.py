"""a210 - Cosine (scaled) attention. NEW: scoring by cosine similarity to a learned query, not gated MLP.

Replaces the Ilse gated scorer with scaled cosine-similarity attention: score_i = s * cos(h_i, q) for a
learned query q and learned temperature s. Cosine geometry is magnitude-invariant (||h|| is grade-
uninformative per the diagnostics), so this scores purely on DIRECTION in feature space -- a different
attention geometry. Concept-free, self-contained, permutation/size-invariant, deterministic.
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
        self.query = nn.Parameter(torch.randn(hidden_dim) * (hidden_dim ** -0.5))
        self.log_s = nn.Parameter(torch.tensor(2.3))        # scale ~10 init
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        s = self.log_s.exp().clamp(max=100.0)
        e = s * (F.normalize(h, dim=1) @ F.normalize(self.query, dim=0))   # [N] cosine score
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
