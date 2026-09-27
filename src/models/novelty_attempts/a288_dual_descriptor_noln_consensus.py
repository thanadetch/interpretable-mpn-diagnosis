"""a288 — a286's dual descriptor (sparse 1.5-entmax ⊕ plain-mean diffuse, concat) fed to a K=4
softmax-weighted consensus head, but WITHOUT the LayerNorm (uni2 seed-2 fish #9; best-of-N, seed-2
only, NOT robust).

Hypothesis being tested: a285/a286 both peaked at best_epoch=1 on every backbone -> the LayerNorm on
the pooled descriptor makes the model near-trivial at init so val peaks immediately and training only
degrades it (a273 already showed LayerNorm-before-pool hurts). a286 (WITH LN) cleared uni2+titan but
dropped virchow2 val. a288 removes the LN to let the model actually train past epoch 1 while keeping
a286's diffuse channel (which lifted the over-concentrated backbones). DISTINCT from a286 (LN present),
a274 (single descriptor, uniform 1/K reads), a283/a285 (single descriptor). Self-contained, perm/size-
invariant, deterministic eval, MPS-safe.
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


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.P = nn.Linear(2 * hidden_dim, K)               # reads the concatenated [sparse; mean]
        self.G = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K))
        self.read_w = nn.Parameter(torch.zeros(K))
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        z_sparse = torch.mv(h.t(), a)
        z_mean = h.mean(dim=0)
        z = torch.cat([z_sparse, z_mean], dim=0)            # NO LayerNorm
        u = self.P(z) * (1.0 + self.G) + self.b
        y = (F.softmax(self.read_w, dim=0) * u).sum().view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
