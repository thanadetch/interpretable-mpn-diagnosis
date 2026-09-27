"""a239 - a215 (learnable-alpha entmax) but alpha INITIALISED LOW (alpha=1.2). Init-sweep probe.

Identical to a215 except alpha_raw starts at -1.386 (sigmoid -> 0.2 -> alpha=1.2, near softmax). Used to
test whether a215's converged alpha~1.5 is a genuine attractor or just an artifact of its 1.5 init: if
a239 (started at 1.2) climbs toward ~1.5 it supports an attractor; if it stays near 1.2 the learnable-alpha
is effectively inert. Concept-free, self-contained, permutation/size-invariant, deterministic, n=1 safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.alpha_raw = nn.Parameter(torch.tensor(-1.3863))   # sigmoid->0.2 -> alpha=1.2 init (LOW)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        alpha = 1.0 + torch.sigmoid(self.alpha_raw)
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
