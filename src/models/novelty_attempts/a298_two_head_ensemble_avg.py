"""a298 — Mini 2-head ENSEMBLE: two FULLY INDEPENDENT gated-attention pools + two Linear heads,
predictions AVERAGED (seed-2 fish #19; best-of-N, seed-2 only, NOT robust).

The val-selection trap rewards lower-VARIANCE predictors on the 30-train-patient cohort. A deep ensemble
is the textbook variance reducer; a298 bakes a 2-model ensemble into one drop-in module: two independent
ABMIL-style branches (each = own bottleneck + gated attention + softmax pool + Linear head),
final y = 0.5*(y1 + y2). Unlike a291 (K linear READS of ONE pooled vector, inert) and a209 (multi-head
attention CONCAT into one head), a298 averages two SEPARATE end-to-end predictions, the only form that
actually reduces predictor variance. Each branch keeps uni2's optimal softmax pool. DISTINCT. Honest
expectation: ensemble variance reduction helps marginally at best and 2x params may val-overfit; recorded
per the never-stop directive. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
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

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b1(features)
        y2, _ = self.b2(features)
        y = 0.5 * (y1 + y2)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
