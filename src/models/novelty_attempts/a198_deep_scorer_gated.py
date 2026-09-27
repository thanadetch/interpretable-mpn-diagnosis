"""a198 - Deeper (2-layer) attention scorer. NEW lever: more expressive PATCH-SCORING function.

Baseline scores each patch with a single gated layer. a198 deepens the scorer to a 2-layer MLP
(V*U -> Linear -> GELU -> Linear -> 1) so the saliency function can be nonlinear in the gated rep,
while keeping the pooling/head identical. Tests whether a richer scoring fn (not a richer pool) helps.
Concept-free, self-contained, deterministic.
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
        self.scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.scorer(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
