"""a195 - Mixture-of-pooling-experts. NEW family: input-dependent gate over 4 pooling operators.

Rather than commit to one pooling, compute four expert poolings (attention-mean, plain mean, max,
logsumexp) and let an input-dependent gate (from a bag descriptor) softly MIX them per bag -- so the
model can read different bags differently (diffuse vs peaky) within one head. Distinct from a188
(which CONCATs fixed statistics); here the gate is learned & bag-adaptive over the pooled vectors.
Concept-free, self-contained, permutation/size-invariant, deterministic.
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
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 4)          # bag-descriptor -> expert weights
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        n = h.shape[0]
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z_attn = torch.mv(h.t(), a)
        z_mean = h.mean(0)
        z_max = h.max(0).values
        z_lse = (torch.logsumexp(h, dim=0) - math.log(n))
        experts = torch.stack([z_attn, z_mean, z_max, z_lse], dim=0)   # [4,D]
        g = F.softmax(self.gate(z_mean), dim=0)                        # [4] bag-adaptive
        z = torch.mv(experts.t(), g)                                   # [D]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
