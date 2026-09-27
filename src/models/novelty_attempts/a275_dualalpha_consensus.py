"""a275 — Dual-alpha pool + DRC consensus head (UNTRIED COMBINATION of a272 + a274).

Combines the two non-trivial levers tried this session: (a272) pool the a215 gated scores at TWO
fixed entmax sparsity levels (alpha=1.2 diffuse + 1.5 sharp) and concat -> z[2H]; (a274) read z
with a K=4 calibrated linear consensus head (uniform mean of K ordinal reads) instead of one Linear.
Leans diffuse (the 1.2 branch) — the principled uni2-friendly direction — with a variance-reduced
readout. Both alphas FIXED (no entropy-conditioning; not a269-killed). Drop-in, perm/size-invariant,
deterministic, MPS-safe. Honest: both parents failed/were-lucky individually; this is a long-shot combo.
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
        self.P = nn.Linear(2 * hidden_dim, K)
        self.G = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K))
        self.ALPHA_LO = 1.2
        self.ALPHA_HI = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a_lo = entmax_bisect(e, self.ALPHA_LO)
        a_hi = entmax_bisect(e, self.ALPHA_HI)
        z = torch.cat([torch.mv(h.t(), a_hi), torch.mv(h.t(), a_lo)], dim=0)   # [2H]
        u = self.P(z) * (1.0 + self.G) + self.b
        y = u.mean().view(-1)
        if return_attention:
            return y, a_hi, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
