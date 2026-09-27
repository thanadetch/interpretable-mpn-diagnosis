"""a295 — Lower-capacity UNGATED attention pool (tests whether REDUCING capacity improves uni2 test
generalisation; seed-2 fish #16; best-of-N, seed-2 only, NOT robust).

Every PERTURBATION of the baseline tested so far either val-overfits (test down) or lands at ~baseline:
sparser pool (a290), more-diffuse pool (a294), consensus head (a291), feature refine (a292/a293) all
ADD capacity or move off the softmax optimum. The one untried DIRECTION is REDUCING capacity: on a
30-train-patient cohort the val-selection trap rewards lower-variance models, so a SIMPLER attention
might generalise better to test. a295 replaces the baseline's GATED attention (V=tanh ⊙ U=sigmoid, then
W) with a plain UNGATED single-layer attention (W·tanh(V·h)) computed in a SMALL 32-d subspace, then the
baseline softmax pool (uni2's optimum) + single Linear head. ~half the attention params of the baseline.
DISTINCT from ABMIL (removes the sigmoid gate + shrinks attention dim). Honest expectation:
lower capacity may not val-overfit but also has no reason to exceed the baseline's test; recorded per the
never-stop directive. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, att_dim=32):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # ungated, low-dim attention: h -> tanh(att_dim) -> scalar score
        self.att_v = nn.Sequential(nn.Linear(hidden_dim, att_dim), nn.Tanh())
        self.att_w = nn.Linear(att_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.att_w(self.att_v(h)).squeeze(-1)             # ungated attention score
        a = F.softmax(e, dim=0)                                # baseline softmax pool (uni2's optimum)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
