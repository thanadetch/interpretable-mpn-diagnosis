"""a303 — Architecturally-DIVERSE 2-branch ensemble: one SOFTMAX-gated branch + one 1.5-ENTMAX branch,
averaged (attacks a298's titan gap via backbone-optimal-mode diversity; seed-2 fish #24; best-of-N,
seed-2 only, NOT robust).

Disjoint-helper facts: ENTMAX (alpha=1.5) is titan's optimal pool (sparse helps the diffuse backbone,
a215 titan ~0.96), SOFTMAX is uni2's optimal pool (a290: uni2 optimum = alpha=1). a298 used two IDENTICAL
softmax branches -> averaging regressed titan. a303 makes the two branches use the TWO backbone-optimal
pooling modes: branch1 = softmax-gated (uni2-optimal), branch2 = 1.5-entmax-gated (titan-optimal), y =
0.5*(y1+y2). Rationale: for titan, the entmax branch stays sharp (~0.96) so the average is held up; for
uni2, the softmax branch carries it; the two modes are decorrelated -> genuine ensemble diversity. This
is the most disjoint-helper-aware ensemble: combine the two modes that are each optimal for a different
backbone. DISTINCT from a298 (identical softmax branches) and all single-pool models. Honest expectation:
each backbone's average is dragged by its non-optimal branch, so may help none fully; recorded per the
never-stop directive. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
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


class _Branch(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, mode):
        super().__init__()
        self.mode = mode    # "softmax" or "entmax"
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0) if self.mode == "softmax" else entmax_bisect(e, 1.5)
        z = torch.mv(h.t(), a)
        return self.classifier(z).view(-1), a


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.b_soft = _Branch(input_dim, hidden_dim, dropout, "softmax")   # uni2-optimal pool
        self.b_ent = _Branch(input_dim, hidden_dim, dropout, "entmax")     # titan-optimal pool

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y1, a1 = self.b_soft(features)
        y2, _ = self.b_ent(features)
        y = 0.5 * (y1 + y2)
        if return_attention:
            return y, a1, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
