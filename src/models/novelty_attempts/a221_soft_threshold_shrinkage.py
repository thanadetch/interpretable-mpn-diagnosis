"""a221 - Soft-threshold (shrinkage) on the pooled embedding. NEW: sparsify the BAG vector, not patches.

After the usual gated-attention pool, apply a learnable soft-threshold (shrinkage) to the pooled bag
embedding: z' = sign(z) * relu(|z| - lambda). This zeroes small/noisy feature dimensions of the bag rep
(denoise in feature space, a lasso-like prior) before the head. lambda init ~0 (= baseline). Distinct
from all patch-space mechanisms. Concept-free, self-contained, deterministic.
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
        self.log_lam = nn.Parameter(torch.tensor(-4.0))      # lambda = softplus ~ 0.018 ~ 0 init
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        lam = F.softplus(self.log_lam)
        z = torch.sign(z) * F.relu(z.abs() - lam)            # soft-threshold shrinkage
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
