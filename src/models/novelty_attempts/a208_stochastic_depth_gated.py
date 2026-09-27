"""a208 - Stochastic-depth gated attention. NEW regulariser: randomly bypass the bottleneck transform.

A second projection block after the bottleneck is applied with stochastic depth: at train it is kept
with prob p (else identity), at eval it is always applied scaled by p (standard stochastic depth).
Regularises the feature transform's effective depth without corrupting feature VALUES (unlike noise) ->
another G3-safe regulariser on the only-titan-passing lever. Concept-free, self-contained, det. at eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, p_keep=0.5):
        super().__init__()
        self.p_keep = p_keep
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        delta = self.block(self.ln(h))
        if self.training:
            if torch.rand(1).item() < self.p_keep:
                h = h + delta
        else:
            h = h + self.p_keep * delta
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
