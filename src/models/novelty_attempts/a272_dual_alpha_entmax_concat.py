"""a272 — Dual-alpha entmax concat pool (UNTRIED INSTANCE of multi-pool-concat).

Builds on a215: same gated-attention scorer, but pool the SAME scores at TWO fixed entmax
sparsity levels — alpha=1.2 (denser, keeps more patches = more diffuse) and alpha=1.5 (sparser,
a215's setting) — and CONCAT the two pooled vectors before the head, so the head sees both a
diffuse-density read and a sharper-focal read of the bag. Distinct from a209 (multi-HEAD attention
concat: different scorers) — here ONE scorer, two SPARSITY levels (multi-alpha axis). Both alphas
are FIXED (no learnable alpha, no entropy-conditioning -> does not rely on the a269-killed
backbone-entropy separation). Respects diffuse density (alpha=1.2 branch is broad). Self-contained,
perm/size-invariant, deterministic, MPS-safe. +hidden_dim params on the head (Linear(2H->1)).
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
        Z = p.sum()
        if Z > 1: tau_lo = tau
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
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        self.ALPHA_LO = 1.2   # denser / more diffuse
        self.ALPHA_HI = 1.5   # a215's sparser setting

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a_lo = entmax_bisect(e, self.ALPHA_LO)
        a_hi = entmax_bisect(e, self.ALPHA_HI)
        z = torch.cat([torch.mv(h.t(), a_hi), torch.mv(h.t(), a_lo)], dim=0)   # [2H]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a_hi, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
