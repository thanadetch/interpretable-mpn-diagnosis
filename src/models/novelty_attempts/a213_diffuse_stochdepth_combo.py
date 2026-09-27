"""a213 - Diffuse-temperature (a158) + stochastic-depth (a208) combo. Motivated: the two virchow2-test passers.

a158 (free diffuse temperature) and a208 (stochastic-depth feature transform) BOTH passed titan GATE1
AND pushed virchow2 TEST >= baseline (a158 0.9549, a208 0.9483) -- but each crashed uni2. They are
orthogonal (one shapes attention, one regularises the transform). a213 stacks them to test whether the
combined effect lifts ALL three backbones (the only path to GATE2). Concept-free, self-contained, det.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_IS1 = math.log(math.expm1(1.0))


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
        self.T_raw = nn.Parameter(torch.tensor(_IS1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        delta = self.block(self.ln(h))
        if self.training:
            if torch.rand(1).item() < self.p_keep: h = h + delta
        else:
            h = h + self.p_keep * delta
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        T = F.softplus(self.T_raw).clamp(min=1e-2)
        a = F.softmax(e / T, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
