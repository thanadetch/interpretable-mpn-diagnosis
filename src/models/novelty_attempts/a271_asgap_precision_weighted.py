"""a271 — ASGAP + per-patch precision weighting (UNTRIED COMBINATION).

Builds on a215: keep the 1.5-entmax gated-attention pool VERBATIM, but among the entmax-SELECTED
patches, additionally reweight by a learned per-patch PRECISION p_i = softplus(Linear(h_i)).
Final weight w_i = (a_i * p_i) renormalised, where a_i is the a215 entmax attention (so the hard
zeros of entmax are preserved -> precision only reranks WITHIN the selected region). a107 explored
precision/inverse-variance pooling STANDALONE (diffuse pool); it was never combined with the entmax
gated-attention selection -> this specific composition is untried. Respects the diffuse-density
grading principle (soft reweight, not hard selection); does NOT rely on backbone attention-entropy
separation (a269). Self-contained, permutation/size-invariant, deterministic eval, MPS-safe.

At init the precision head outputs ~constant (Linear init small) so w ~= a -> a271 starts == a215.
+129 params vs a215 (one Linear(128->1)).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.precision = nn.Linear(hidden_dim, 1)   # per-patch log-precision; init small -> ~uniform
        nn.init.zeros_(self.precision.weight); nn.init.zeros_(self.precision.bias)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)                       # [N] sparse entmax attention (hard zeros)
        p = F.softplus(self.precision(h).squeeze(-1))          # [N] positive precision (~1 at init)
        w = a * p                                              # reweight within the selected region
        w = w / w.sum().clamp(min=1e-8)                        # renormalise -> sums to 1, keeps a's zeros
        z = torch.mv(h.t(), w)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
