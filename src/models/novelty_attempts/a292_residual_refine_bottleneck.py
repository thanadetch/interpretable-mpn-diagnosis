"""a292 — Pre-pool RESIDUAL feature-refinement bottleneck, with uni2's PROVEN-optimal softmax pool +
single Linear head kept untouched (seed-2 fish #13; best-of-N, seed-2 only, NOT robust).

The a290/a291 two-axis proof closed the POOLING axis (uni2 optimum = softmax/alpha=1) and the HEAD axis
(uni2 optimum = single Linear) for uni2. The one remaining drop-in lever is the BOTTLENECK / pre-pool
feature transform. a292 keeps the baseline's softmax gated pool + single Linear head EXACTLY, and only
adds a residual refinement block r(.) on the bottleneck features (h <- h + r(h)) with the block's last
layer ZERO-INITIALISED, so at init r==0 -> h unchanged -> a292 == baseline. This isolates the question
"does richer pre-pool feature refinement help uni2 beyond the baseline, without touching pool or head?".
NO entmax (a290: hurts uni2), NO consensus head (a291: hurts uni2), NO LayerNorm (a288: near-init
artifact), NO learnable mixing knob (inert x4). Honest expectation: extra capacity tends to val-overfit
on the 30-train-patient cohort, so likely <= baseline; recorded for completeness per the never-stop
directive. Self-contained, perm/size-invariant, deterministic eval, MPS-safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # residual refinement block; last layer zero-init -> identity at init -> == baseline at init
        self.refine_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.refine_fc2 = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.refine_fc2.weight); nn.init.zeros_(self.refine_fc2.bias)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        h = h + self.refine_fc2(F.gelu(self.refine_fc1(h)))   # residual refinement (identity at init)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)                                # baseline softmax pool (uni2's optimum)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
