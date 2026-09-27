"""a306 — ASGAP at α=2.0 (sparsemax, the sparsest entmax extreme); maps the full titan sparsity axis
(seed-2 fish #27; best-of-N, seed-2 only, NOT robust; also a thesis characterization data point).

The disjoint-helper says titan (diffuse backbone) benefits from sparser pooling. We have the α-curve at
α=1.0 (softmax = baseline), 1.3 (a290), 1.5 (a215). a306 adds the SPARSEST extreme α=2.0 = sparsemax
(Martins & Astudillo 2016; the α→2 limit of Tsallis entmax) to test whether even-sparser-than-1.5 helps
or hurts titan. Bare a215 architecture (gated attention + single Linear head), only α changed to 2.0.
This completes the titan sparsity-optimum curve {1.0, 1.3, 1.5, 2.0} for the thesis characterization of
the disjoint-helper, AND probes whether α=2.0 clears titan's gate. NO ensemble/LayerNorm/learnable-mixing.
Honest expectation: α=2.0 likely over-sparsifies (keeps too few patches → loses diffuse density), so ≤
a215; recorded per the never-stop directive + as the α-curve endpoint. Self-contained, perm/size-invariant,
deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


def entmax_bisect(z, alpha, n_iter=25):
    # generic alpha-entmax via bisection (Peters et al. 2019); alpha=2 -> sparsemax
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
        self.ALPHA = 2.0   # sparsemax (sparsest extreme)

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
