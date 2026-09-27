"""a177 — Diffuse-only temperature (T>=1) + learnable coverage floor (two holistic knobs, both safe).

Composition of the two principled holistic mechanisms, each constrained so it can only express
MORE-holistic, never spike:
  - a176 diffuse-only temperature: T = 1 + softplus(.)  (T>=1, flatten-or-neutral, never sharpen)
  - a171 coverage floor: w = (1-eps)*softmax(e/T) + eps*(1/N)  (guarantee every patch a min weight)
Both encode "grade = diffuse holistic density" and both init to the baseline (T~1, eps~0), so each is
earned. Unlike a171 (free T) this cannot sharpen below 1 -> avoids the uni2 over-sharpen failure.
Self-contained, 2 DOF, deterministic at inference, permutation/size-invariant.
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
        self.T_excess_raw = nn.Parameter(torch.tensor(-6.0))   # T = 1 + softplus -> ~1.0
        self.eps_raw = nn.Parameter(torch.tensor(-6.0))        # eps = sigmoid -> ~0.0025

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        n = e.shape[0]
        T = 1.0 + F.softplus(self.T_excess_raw)
        a = F.softmax(e / T, dim=0)
        eps = torch.sigmoid(self.eps_raw)
        w = (1.0 - eps) * a + eps * (1.0 / n)
        z = torch.mm(w.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
