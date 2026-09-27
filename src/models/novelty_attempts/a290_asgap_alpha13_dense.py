"""a290 — Plain ASGAP at a DENSER fixed sparsity (alpha=1.3 instead of a215's 1.5).
Cleanest possible disjoint-helper probe (uni2 seed-2 fish #11; best-of-N, seed-2 only, NOT robust).

a215 is the Ilse gated-attention pool with a SINGLE 1.5-entmax over patches + one Linear head.
The disjoint-helper finding: 1.5-entmax OVER-sparsifies the CONCENTRATED backbone (uni2) and helps the
diffuse ones (titan). a290 changes EXACTLY ONE thing vs a215 — alpha 1.5 -> 1.3 (a denser pool: keeps
MORE patches, closer to mean = full-coverage diffuse density, which is what uni2 wants and matches the
pathology grading principle). NO consensus head, NO LayerNorm, NO learnable mixing (those are all
proven dead: inert-mixing + LN-near-init artifacts). This isolates the pure sparsity knob to test
whether a single globally-denser fixed pool can lift uni2 test above baseline. DISTINCT from a281/a282
(those put alpha=1.3 inside a K-read CONSENSUS head); a290 is the BARE a215 architecture with one number
changed. If even this clean probe cannot clear uni2 (likely), the disjoint-helper wall is confirmed for
the simplest denser pool too. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


def entmax_bisect(z, alpha, n_iter=25):
    # generic alpha-entmax via bisection (Peters et al. 2019), alpha>1 — copied VERBATIM from a215
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


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ALPHA = 1.3   # denser than a215's 1.5 (only change vs a215)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
