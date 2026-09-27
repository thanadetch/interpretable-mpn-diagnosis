"""a158 — Diffuse-temperature gated attention (holistic-density prior on the strong gated head).

LIGHT, self-contained. Keeps ABMIL exactly (the head that already works well) and adds
ONE learnable scalar: an attention TEMPERATURE T. The grading prior is the pathology principle
that grade = DIFFUSE, bag-wide fibre density (not a few standout patches) — so the attention is
allowed to SPREAD (T>1 → flatter, more mean-like / holistic) rather than spike. T=softplus(T_raw),
init 1.0 = exact baseline; the optimiser raises T only if a more diffuse read generalises better.
No axis / no external file needed. 1 extra DOF over the baseline.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        # num_classes=1 -> scalar regression; num_classes=4 -> multi-class. The diffuse-temperature
        # mechanism is formulation-agnostic; only the final head width changes.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.T_raw = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))  # T=softplus(T_raw)=1.0 init

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        T = F.softplus(self.T_raw).clamp(min=1e-2)
        attn = F.softmax(e / T, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
