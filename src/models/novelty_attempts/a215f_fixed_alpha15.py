"""ASGAP (fixed-alpha variant) — a215 with the learnable alpha REMOVED, alpha fixed at 1.5.

Identical to a215 (Ilse gated attention + 1.5-entmax pool, Peters et al. 2019) in every way EXCEPT
that alpha is a FIXED hyperparameter = 1.5 instead of the learnable `alpha_raw` parameter. a215's
learnable alpha was proven empirically INERT (init-sweep a239/a240: it stays ~1.5), so removing it
should reproduce a215's results within noise while making the code match the thesis claim exactly:
"1.5-entmax with fixed alpha = 1.5" — no learnable-but-inert caveat needed. Verification module: trained
at seed-2 on all 3 backbones and compared to a215 to confirm the reported a215 numbers still hold.
Concept-free, self-contained, permutation/size-invariant, deterministic eval, MPS-safe. One FEWER
parameter than a215 (no alpha_raw scalar).
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
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ALPHA = 1.5   # FIXED hyperparameter (no learnable alpha_raw)

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
