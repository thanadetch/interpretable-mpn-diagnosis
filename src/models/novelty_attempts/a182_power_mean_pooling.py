"""a182 — Generalized (power) mean pooling. NEW family: interpolates mean<->max via a learnable power.

z_j = (Σ_i a_i h_ij^p)^(1/p)   over patches, per feature j  (h>=0 from the ReLU bottleneck).
p=1 -> weighted arithmetic mean (~baseline); p->inf -> max; p->0 -> geometric mean; p<0 -> min-like.
A single learnable p lets the model choose how "peaky" the per-feature aggregation is, a different knob
from attention temperature (that reweights patches; this reshapes the aggregation NONLINEARLY per
feature). Concept-free, self-contained, permutation/size-invariant, deterministic at inference.
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
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.p_raw = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413)=1.0 -> p=1 init (= weighted mean)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # >=0
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        p = F.softplus(self.p_raw).clamp(min=0.2, max=8.0)
        hp = (h + 1e-6).pow(p)                              # [N,hidden]
        z = (torch.mv(hp.t(), a)).clamp(min=1e-9).pow(1.0 / p)   # generalized mean per feature
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
