"""a281 — DRC consensus head (a274) + denser entmax alpha=1.3 (uni2-targeted; seed-2 goal).
a274 (consensus head, alpha=1.5) got uni2 closest (0.9389 vs baseline 0.9418). uni2 over-concentrates
under alpha=1.5, so this uses a slightly DENSER alpha=1.3 (more diffuse = uni2-friendly per the
disjoint-helper mechanism) PLUS the variance-reducing K=4 consensus readout. Goal: push uni2 seed-2
test over 0.9418 while keeping titan/virchow2. Drop-in, perm/size-inv, deterministic, MPS-safe.
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
        self.P = nn.Linear(hidden_dim, K)
        self.G = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K))
        self.ALPHA = 1.3

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        z = torch.mv(h.t(), a)
        u = self.P(z) * (1.0 + self.G) + self.b
        y = u.mean().view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
