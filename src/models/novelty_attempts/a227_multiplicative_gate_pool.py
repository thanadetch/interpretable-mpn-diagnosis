"""a227 - Multiplicative instance-gate pooling. NEW: sigmoid per-patch GATE (sum, not softmax-normalised).

Each patch gets an independent sigmoid keep-gate g_i in (0,1); pool = (sum_i g_i h_i)/(sum_i g_i). Unlike
softmax attention (competitive, sums to 1), gates are INDEPENDENT (a patch's weight does not depend on
others) -> a non-competitive saliency that can keep many or few patches without a fixed budget.
Concept-free, self-contained, permutation/size-invariant, deterministic.
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
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        g = torch.sigmoid(self.gate(h).squeeze(-1))          # [N] independent keep-gates
        z = torch.mv(h.t(), g) / g.sum().clamp(min=1e-6)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, g / g.sum().clamp(min=1e-6), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
