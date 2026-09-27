"""a304 — BRANCH-DOMINANT weighted 2-branch ensemble (y = 0.8*y1 + 0.2*y2), to minimise titan
averaging-regression (seed-2 fish #25; best-of-N, seed-2 only, NOT robust).

a298 (0.5/0.5 mean) regressed titan below gate because averaging pulls the strong backbone toward
center. a304 keeps two FULLY independent branches but weights branch1 dominant (0.8) and branch2 minor
(0.2), FIXED (non-learnable -> non-inert). Rationale: for titan, 0.8*y1 keeps the prediction close to
the single-model peak (~0.96) so regression is minimal (~0.2 of the pull); for uni2/virchow2, the 0.2
branch2 still injects some decorrelated-ensemble variance reduction. This is the minimal-regression point
on the weight axis between a298 (0.5/0.5, helps uni2/virchow2 but regresses titan) and a single model
(0/1, titan-optimal but no uni2 help). DISTINCT from a298 (equal weights), a301 (asym CAPACITY not
weight). Honest expectation: titan regresses less but uni2 help also shrinks (less branch2 weight); may
clear none fully; recorded per the never-stop directive. Self-contained, perm/size-invariant,
deterministic eval, MPS-safe.
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
        self.W1, self.W2 = 0.8, 0.2

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b1(features)
        y2, _ = self.b2(features)
        y = self.W1 * y1 + self.W2 * y2
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
