"""a274 / DRC-ASGAP — Deterministic Rank-Consensus head over the a215 entmax pool.

Round-2 gap-first workflow's #1 ("least-dead") candidate. Keep a215's 1.5-entmax gated pool VERBATIM;
replace the single Linear(128->1) head with the uniform 1/K mean of K=4 linear ordinal reads of the
SAME pooled vector z, each with its own learned direction + gain — a variance-reducing consensus of
correlated reads instead of one fragile projection. Non-rehash (verified vs a251 monotone-CDF, a40
seed-fuse, a209 multihead-pool). alpha fixed 1.5 (not learnable). Drop-in, perm/size-invariant,
deterministic, MPS-safe. Honest: on diffuse attention this degenerates toward a calibrated Linear
head (likely ties a215); it is a variance-reduction probe, not a credible GATE-clearer.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


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


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.P = nn.Linear(hidden_dim, K)          # K ordinal reads of z
        self.G = nn.Parameter(torch.zeros(K))      # per-read gain (init 0 -> gain 1)
        self.b = nn.Parameter(torch.zeros(K))      # per-read bias
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        z = torch.mv(h.t(), a)
        u = self.P(z) * (1.0 + self.G) + self.b    # [K] K calibrated ordinal reads
        y = u.mean().view(-1)                      # uniform consensus
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
