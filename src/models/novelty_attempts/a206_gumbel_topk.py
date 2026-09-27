"""a206 - Gumbel-softmax hard patch selection. NEW: stochastic HARD subset at train, top-set at eval.

Differentiable hard selection: at train, sample patch keep-gates via Gumbel-softmax (straight-through)
so the bag is pooled over a stochastic hard subset (regularises against over-reliance on fixed patches);
at eval, deterministically keep patches whose gate prob > 0.5 and mean-pool. Distinct from sparsemax
(deterministic sparse weights) and from dropout (this is a learned per-patch keep distribution).
Concept-free, self-contained, deterministic at inference.
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
        self.gate = nn.Linear(hidden_dim, 1)            # keep-logit per patch
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        g = self.gate(h).squeeze(-1)                    # [N] keep logit
        if self.training:
            u = torch.rand_like(g).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            keep = torch.sigmoid((g + gumbel) / 0.5)    # soft keep prob (ST relaxation)
        else:
            keep = (torch.sigmoid(g) > 0.5).float()
            if keep.sum() < 1: keep = torch.ones_like(g)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0) * keep
        a = a / a.sum().clamp(min=1e-8)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
