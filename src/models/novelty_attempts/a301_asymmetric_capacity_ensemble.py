"""a301 — ASYMMETRIC-capacity 2-branch ensemble (threads a298's titan gap; seed-2 fish #22; best-of-N,
seed-2 only, NOT robust).

a298 (two IDENTICAL full branches, averaged) cleared uni2+virchow2 but the averaging regressed titan
below gate. a300 (anti-shrink) and a299 (shared bottleneck) both failed to fix it. The remaining idea:
make the two branches ASYMMETRIC in capacity — branch1 = the full baseline gated-attention model;
branch2 = a LOW-capacity ungated branch. On the clean titan backbone both branches should converge to
the same good solution (AGREE -> minimal averaging-regression on titan), while on the noisier uni2
backbone the low-cap branch behaves differently (DIVERSITY -> variance-reduction help). final y =
0.5*(y1+y2). DISTINCT from a298 (symmetric full branches), a299 (shared bottleneck), a300 (anti-shrink).
Honest expectation: titan's gate is near the single-model ceiling so even reduced regression may not
cross it; recorded per the never-stop directive. Self-contained, perm/size-invariant, deterministic
eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _FullBranch(nn.Module):
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


class _LowCapBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, att_dim=32):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.att_v = nn.Sequential(nn.Linear(hidden_dim, att_dim), nn.Tanh())   # ungated, low-dim
        self.att_w = nn.Linear(att_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        h = self.bottleneck(features)
        e = self.att_w(self.att_v(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        return self.classifier(z).view(-1), a


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.b1 = _FullBranch(input_dim, hidden_dim, dropout)
        self.b2 = _LowCapBranch(input_dim, hidden_dim, dropout)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b1(features)
        y2, _ = self.b2(features)
        y = 0.5 * (y1 + y2)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
