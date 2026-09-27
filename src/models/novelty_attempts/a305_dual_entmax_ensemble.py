"""a305 — 2-branch ensemble where BOTH branches use 1.5-ENTMAX (titan-optimal pool), mean-averaged
(makes the trifecta impossibility concrete; seed-2 fish #26; best-of-N, seed-2 only, NOT robust).

a298 (two SOFTMAX branches) helped uni2 (softmax = uni2's optimum) but regressed titan. a305 flips it:
both branches use 1.5-ENTMAX (titan's optimum). Prediction: for TITAN both branches agree on sharp
attention (~0.96) so the average stays HIGH (minimal regression) -> titan may clear; but for UNI2 entmax
is suboptimal (a290: uni2 wants alpha=1) so both branches are ~0.926 and uni2 FAILS. This is the cleanest
single demonstration that NO single model / same-type ensemble satisfies both backbones: a softmax-ensemble
(a298) clears uni2 not titan, an entmax-ensemble (a305) clears titan not uni2 -> the disjoint-helper is a
hard per-backbone pooling-optimum conflict. DISTINCT from a298 (softmax branches) and a303 (mixed). Honest
expectation: titan up, uni2 down -> still no GATE2; recorded per the never-stop directive + as the concrete
impossibility illustration. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha, n_iter=25):
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max() - (1.0 / am1) * 0.0
    lo = z.max() - 1.0
    tau_lo = lo; tau_hi = hi
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        if p.sum() > 1: tau_lo = tau
        else: tau_hi = tau
    p = torch.clamp(z - tau_hi, min=0) ** (1.0 / am1)
    return p / p.sum().clamp(min=1e-8)


class _EntmaxBranch(nn.Module):
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
        a = entmax_bisect(e, 1.5)
        z = torch.mv(h.t(), a)
        return self.classifier(z).view(-1), a


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.b1 = _EntmaxBranch(input_dim, hidden_dim, dropout)
        self.b2 = _EntmaxBranch(input_dim, hidden_dim, dropout)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b1(features)
        y2, _ = self.b2(features)
        y = 0.5 * (y1 + y2)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
