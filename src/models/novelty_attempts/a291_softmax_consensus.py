"""a291 — Baseline SOFTMAX gated pool (uni2's PROVEN optimal sparsity, alpha=1) + K=4 learned-weight
consensus head (uni2 seed-2 fish #12; best-of-N, seed-2 only, NOT robust).

Motivated directly by a290: uni2's pooling-sparsity OPTIMUM is alpha=1 = plain softmax = the baseline
(any entmax alpha>1 underperforms softmax on the concentrated uni2 backbone). So instead of fighting
that with sparse entmax, a291 KEEPS the baseline's softmax pool (Ilse gated attention -> softmax over
patches -> attention-weighted mean) UNCHANGED, and only replaces the single Linear head with the a274
consensus head (K=4 independent linear ordinal reads of the SAME pooled z, combined by a learned-softmax
read_w). This tests whether read-variance-reduction adds anything OVER the baseline when it is NOT
fighting the sparsity optimum. DISTINCT from a274/a283/a285/a289 (all used ENTMAX pooling); a291 is the
first consensus-head variant on the SOFTMAX (baseline) pool. No LayerNorm (avoids the near-init artifact).
At init read_w=0 -> uniform 1/K consensus -> a291 ~= baseline with a K-averaged head. Self-contained,
perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.read_dir = nn.Parameter(torch.zeros(K, hidden_dim))
        nn.init.xavier_uniform_(self.read_dir)
        self.read_gain = nn.Parameter(torch.ones(K))
        self.read_b = nn.Parameter(torch.zeros(K))
        self.read_w = nn.Parameter(torch.zeros(K))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)                              # baseline softmax pool (alpha=1, uni2's optimum)
        z = torch.mv(h.t(), a)                              # attention-weighted mean
        u = (self.read_dir * z).sum(dim=1) * self.read_gain + self.read_b   # [K] independent reads of z
        y = (F.softmax(self.read_w, dim=0) * u).sum().view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
