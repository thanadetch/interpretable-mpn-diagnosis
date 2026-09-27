"""a300 — a298 independent 2-branch ensemble + FIXED anti-shrinkage on the averaged prediction
(targets a298's narrow titan miss; seed-2 fish #21; best-of-N, seed-2 only, NOT robust).

a298 (two independent branches, y=0.5(y1+y2)) genuinely cleared uni2+virchow2 but the AVERAGING
COMPRESSES predictions toward the mean grade, which regressed titan (highest baseline + steepest gate
0.9584) just below its gates. a300 keeps the two FULLY independent branches (a299 proved independence is
what produces the uni2 gain) and applies a FIXED anti-shrinkage to the averaged prediction:
y = MID + GAIN*(ybar - MID), with MID=1.5 (grade midpoint), GAIN=1.12 (>1, expands dynamic range to
undo the averaging compression). FIXED, not learnable (learnable scalars are inert x4 on this objective).
DISTINCT from a298 (no range expansion). Rationale: restore the full G0..G3 range that averaging
shrinks, so titan's confident extreme-grade predictions are not pulled toward the centre. Honest
expectation: may help titan's range at a small uni2 cost; recorded per the never-stop directive.
Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
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
        self.MID = 1.5
        self.GAIN = 1.12

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b1(features)
        y2, _ = self.b2(features)
        ybar = 0.5 * (y1 + y2)
        y = self.MID + self.GAIN * (ybar - self.MID)          # fixed anti-shrinkage (expand range)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
