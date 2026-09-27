"""a202 - DropConnect on the classifier head. NEW regulariser: noise the HEAD weights, not features.

Targets the same titan-GATE1 regularisation lever as a172/a167 but at the safest possible site: random
weight-masking (DropConnect) on the FINAL classifier only, at train. The patch features, attention and
pooled embedding are untouched -> G3's high-density signal is fully preserved (the GATE2 killer), while
the head is regularised against val-overfit. Off at inference (deterministic). Self-contained, concept-free.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, p_dc=0.2):
        super().__init__()
        self.p_dc = p_dc
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        W = self.classifier.weight
        if self.training and self.p_dc > 0:
            W = W * (torch.rand_like(W) >= self.p_dc).float() / (1.0 - self.p_dc)
        y = (F.linear(z, W, self.classifier.bias)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
