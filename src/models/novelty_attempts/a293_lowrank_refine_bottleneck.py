"""a293 — LOW-RANK (rank-16) residual feature refinement, targeting a292's specific failure mode
(seed-2 fish #14; best-of-N, seed-2 only, NOT robust).

a292 added a FULL-rank (128->128) residual refinement on the bottleneck and val-overfit (uni2 val 0.7939
PASS but test 0.9096 FAIL, best_epoch=1 — too much capacity on 30 train patients). a293 keeps the exact
same idea (baseline softmax pool + single Linear head untouched = uni2's proven optima; zero-init residual
so == baseline at init) but throttles the refinement to a LOW-RANK bottleneck (128 -> 16 -> 128), ~8x
fewer refine params, to test whether a MINIMAL feature refinement can help uni2 without the capacity
val-overfit. DISTINCT from a292 (full-rank refine). NO entmax/consensus/LayerNorm/learnable-mixing (all
proven dead). Honest expectation: still likely <= baseline (the 3-axis proof says baseline is uni2's
optimum), but the smallest-capacity refinement is the least-overfit-prone untested point. Self-contained,
perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, rank=16):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # low-rank residual refinement: hidden -> rank -> hidden, last layer zero-init (identity at init)
        self.refine_down = nn.Linear(hidden_dim, rank)
        self.refine_up = nn.Linear(rank, hidden_dim)
        nn.init.zeros_(self.refine_up.weight); nn.init.zeros_(self.refine_up.bias)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        h = h + self.refine_up(F.gelu(self.refine_down(h)))   # low-rank residual (identity at init)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)                                # baseline softmax pool (uni2's optimum)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
