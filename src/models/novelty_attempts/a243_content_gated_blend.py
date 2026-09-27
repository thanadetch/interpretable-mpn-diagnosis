"""a243 - Content-gated mean/attention blend. NEW: per-bag gate from content (not a global scalar).

a237 (global learnable lambda blend) would converge inert like every scalar knob this session. a243 makes
the attention-vs-mean mix a PER-BAG gate computed from the bag content: g = sigmoid(MLP(mean(h))), then
z = g*z_attn + (1-g)*z_mean. A bag whose attention over-concentrates (e.g. uni2) can route itself toward
the robust mean read, while bags that benefit from focus keep attention -- decided per bag, structurally,
from content. Targets disjoint-helper + val-variance at once. Concept-free, self-contained,
permutation/size-invariant, deterministic, n=1 safe.
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
        self.gate = nn.Sequential(                           # bag content -> per-bag attn/mean mix
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        z_mean = h.mean(dim=0)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z_attn = torch.mv(h.t(), a)
        g = torch.sigmoid(self.gate(z_mean))                 # per-bag gate from content [1]
        z = g * z_attn + (1.0 - g) * z_mean
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
