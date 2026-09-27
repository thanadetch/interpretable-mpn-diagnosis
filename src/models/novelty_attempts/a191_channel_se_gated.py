"""a191 - Squeeze-Excite channel recalibration + gated attention. NEW lever: per-FEATURE gating.

A squeeze-excite block computes a global bag descriptor (mean over patches) and produces a per-CHANNEL
gate that recalibrates the bottleneck feature dimensions BEFORE the usual gated-attention pool. Every
prior candidate reweights PATCHES; this reweights FEATURE CHANNELS (which fibre-pattern dimensions
matter for this bag). Init gate ~1 (near baseline). Concept-free, self-contained, permutation/size-
invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, r=4):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.se1 = nn.Linear(hidden_dim, hidden_dim // r)
        self.se2 = nn.Linear(hidden_dim // r, hidden_dim)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        desc = h.mean(dim=0)                                # [D] squeeze
        gate = torch.sigmoid(self.se2(F.relu(self.se1(desc))))  # [D] excite (channel gate)
        h = h * gate.unsqueeze(0)                           # recalibrate channels
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
