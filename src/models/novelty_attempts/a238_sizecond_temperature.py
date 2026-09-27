"""a238 - Bag-size-conditioned attention temperature. NEW lever: scale sharpness with patch count.

Bag size varies widely (N ~ 18-113 patches per ROI). a238 lets the attention temperature depend on the
(log) bag size: T = softplus(t0 + t1 * log N), so the model can read small bags sharper and large bags
more diffuse (or vice versa) -- a size-adaptive sharpness. Distinct from a170 (entropy-conditioned T).
If t1 -> 0 it reduces to a single global temperature. Concept-free, self-contained, permutation-invariant,
deterministic, n=1 safe (log 1 = 0).
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
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
        self.t0 = nn.Parameter(torch.tensor(0.5413))         # softplus(0.5413)+0.2 ~ 1.2 initial T
        self.t1 = nn.Parameter(torch.tensor(0.0))            # size-coupling, starts at 0 (global T)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        logN = math.log(max(h.shape[0], 1))
        T = F.softplus(self.t0 + self.t1 * logN) + 0.2
        a = F.softmax(e / T, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
