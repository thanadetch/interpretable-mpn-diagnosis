"""a302 — 3-branch independent ensemble combined by the MEDIAN (not the mean) of the 3 predictions
(targets a298's titan gap with a non-regressing combine; seed-2 fish #23; best-of-N, seed-2 only,
NOT robust).

Across the ensemble family the MEAN combine regresses the strong titan backbone toward the centre
(a298 0.9548, a300 0.9499, a301 0.9415, all < gate 0.9584), because any diversity that helps the noisy
uni2 drags titan's mean down. The MEDIAN is the one combine that does NOT regress a backbone where the
branches agree: for titan (3 branches converge to ~0.958) median ≈ each branch (no averaging-pull),
while for uni2 (branches diverge) the median is a robust central estimate (variance reduction, rejects
one outlier branch). a302 = three FULLY independent ABMIL-style branches; y = median(y1,y2,y3).
The median of 3 scalars is differentiable (it selects the middle branch; gradient flows there).
DISTINCT from a298 (mean of 2), a300 (anti-shrink), a301 (asym mean). Honest expectation: median may
preserve titan better but its gate is near the single-model ceiling; recorded per the never-stop
directive. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Branch(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        return self.classifier(z).view(-1), a


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.b1 = _Branch(input_dim, hidden_dim, dropout)
        self.b2 = _Branch(input_dim, hidden_dim, dropout)
        self.b3 = _Branch(input_dim, hidden_dim, dropout)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b1(features)
        y2, _ = self.b2(features)
        y3, _ = self.b3(features)
        y = torch.median(torch.stack([y1, y2, y3], dim=0), dim=0).values   # median of the 3 predictions
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
