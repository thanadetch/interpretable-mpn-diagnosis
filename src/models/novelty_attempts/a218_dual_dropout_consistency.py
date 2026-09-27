"""a218 - Dual-dropout averaged pooling. NEW: average two independently-dropped-out views of the SAME head.

At train, run the SAME gated head twice with independent dropout masks and average the two pooled
vectors (an R-Drop-style implicit consistency / variance reduction within one forward). At eval (dropout
off) both views are identical -> baseline-equivalent architecture but trained toward dropout-invariant
predictions. Concept-free, self-contained, deterministic at inference.
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

    def _pool(self, features):
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        return torch.mv(h.t(), a)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if self.training:
            z = 0.5 * (self._pool(features) + self._pool(features))   # two dropout views
        else:
            z = self._pool(features)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
