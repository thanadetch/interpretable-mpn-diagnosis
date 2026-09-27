"""a289 — Multi-alpha entmax consensus pool (a274 consensus-head family; uni2 seed-2 fish #10;
best-of-N, seed-2 only, NOT robust).

Mechanism (targets the disjoint-helper directly): a215's single 1.5-entmax pool over-sparsifies the
CONCENTRATED backbone (uni2). Here the SAME gated-attention scores e are pooled at K=4 FIXED sparsity
levels alpha = [1.1, 1.25, 1.4, 1.55] (denser -> sparser), giving K pooled descriptors z_k. Each z_k
gets its own learned linear ordinal read (independent direction W_k + gain), and the K reads are
combined by a learned softmax consensus weight read_w. The model can GLOBALLY up-weight the denser-alpha
reads (closer to mean pooling = full coverage) for the concentrated backbone, and the sparser reads for
the diffuse backbones — WITHOUT any per-bag attention-entropy conditioning (the alpha-mix is a learned
global parameter, not data-routed) and WITHOUT LayerNorm (so no near-init epoch-1 artifact like a285/a286).
At init read_w=0 -> uniform 1/K consensus of the K alpha-reads.

DISTINCT from a272 (only 2 alphas, concat -> single Linear head) and a274/a283 (single alpha=1.5, K reads
of ONE pooled z). Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
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
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.ALPHAS = (1.1, 1.25, 1.4, 1.55)               # K=4 sparsity levels, denser -> sparser
        K = len(self.ALPHAS)
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.read_dir = nn.Parameter(torch.zeros(K, hidden_dim))   # per-alpha read direction
        nn.init.xavier_uniform_(self.read_dir)
        self.read_gain = nn.Parameter(torch.ones(K))
        self.read_b = nn.Parameter(torch.zeros(K))
        self.read_w = nn.Parameter(torch.zeros(K))                 # consensus weights (softmax)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        zs = []
        a_last = None
        for alpha in self.ALPHAS:
            a = entmax_bisect(e, alpha)
            a_last = a
            zs.append(torch.mv(h.t(), a))                  # [H]
        Z = torch.stack(zs, dim=0)                          # [K, H]
        u = (Z * self.read_dir).sum(dim=1) * self.read_gain + self.read_b   # [K] independent reads
        y = (F.softmax(self.read_w, dim=0) * u).sum().view(-1)
        if return_attention:
            return y, a_last, None                          # return the sparsest attention for viz
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
