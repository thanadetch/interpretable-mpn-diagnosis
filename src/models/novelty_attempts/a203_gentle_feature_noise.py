"""a203 - GENTLE fixed feature noise (sigma=0.04). NEW point on the only-titan-passing lever.

a172's learnable noise landed ~0.1 and crashed G3 on weak backbones. a203 pins a much SMALLER fixed
sigma=0.04 -> just enough denoising regularisation to (maybe) keep the titan-GATE1 gain while being too
gentle to wipe the high-density G3 signal. Train-only, off at inference. Self-contained, concept-free.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, sigma=0.04):
        super().__init__()
        self.sigma = sigma
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        if self.training and self.sigma > 0:
            h = h + self.sigma * torch.randn_like(h)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
