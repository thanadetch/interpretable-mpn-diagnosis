"""a196 - Learnable-prototype distance ORDINAL head. NEW family: distance-to-grade-prototype, not linear.

Pool with the usual gated attention, then predict the grade as the soft-distance-weighted expectation
over FOUR learnable grade prototypes (G0..G3) in the pooled embedding space:
    d_k = ||z - p_k||^2 ;  w = softmax(-d_k / tau) ;  y = Σ_k w_k * k
This is an ordinal, metric-learning head (a different decision geometry than the baseline Linear(128,1))
that ties the scalar output to learned class anchors. Self-contained (prototypes are parameters, no
external file), concept-free, deterministic, permutation/size-invariant.
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
        self.prototypes = nn.Parameter(torch.randn(4, hidden_dim) * (hidden_dim ** -0.5))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("grades", torch.tensor([0.0, 1.0, 2.0, 3.0]))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)                                  # [D]
        d = ((z.unsqueeze(0) - self.prototypes) ** 2).sum(dim=1)   # [4]
        tau = self.log_tau.exp().clamp(min=1e-2, max=50.0)
        w = F.softmax(-d / tau, dim=0)                          # [4]
        y = (w * self.grades).sum().view(-1)                    # expected grade
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
