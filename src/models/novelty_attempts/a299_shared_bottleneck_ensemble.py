"""a299 — SHARED-bottleneck 2-branch ensemble (threads a298's bias-variance trade-off; seed-2 fish #20;
best-of-N, seed-2 only, NOT robust).

a298 (two FULLY independent branches, averaged) genuinely cleared uni2 (+0.0154, variance reduction) but
its averaging-BIAS regressed the already-strong titan below gate (0.9548<0.9584). a299 keeps the ensemble
idea but SHARES the bottleneck (feature extractor) across the two branches, giving each its own gated
attention + Linear head only. Rationale: shared features make the two branches more CORRELATED on the
backbone where the single model is already strong (titan) -> less averaging-bias there, while still
offering some decorrelated-head variance reduction for the noisy backbone (uni2). Fewer params than a298
(one bottleneck, not two) -> less val-overfit. final y = 0.5*(y1+y2). DISTINCT from a298 (independent
bottlenecks) and a209 (concat). Honest expectation: a compromise that may help neither fully; recorded
per the never-stop directive. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        return self.classifier(z).view(-1), a


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(                       # SHARED feature extractor
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.h1 = _Head(hidden_dim)
        self.h2 = _Head(hidden_dim)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        y1, a1 = self.h1(h)
        y2, _ = self.h2(h)
        y = 0.5 * (y1 + y2)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
