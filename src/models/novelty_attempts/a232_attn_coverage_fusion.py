"""a232 - Fusion of attention-mean and coverage-histogram. Ablation companion to a231.

Hedges the salient read (Ilse gated attention-weighted mean) WITH the diffuse coverage profile (soft
histogram over a learned density axis, as in a231). Concat both and grade from the union, so the head can
use peak-salient evidence and bag-wide density distribution together. Concept-free, self-contained,
permutation/size-invariant, deterministic, n=1 safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_bins=12):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.density = nn.Linear(hidden_dim, 1)
        self.register_buffer("centers", torch.linspace(-2.0, 2.0, n_bins))
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(hidden_dim + n_bins, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z_attn = torch.mv(h.t(), a)                          # salient read [H]
        s = self.density(h).squeeze(-1)
        s = (s - s.mean()) / (s.std(unbiased=False) + 1e-5)
        tau = F.softplus(self.log_tau) + 0.2
        hist = F.softmax(-(s[:, None] - self.centers[None, :]) ** 2 / tau, dim=1).mean(dim=0)  # coverage [K]
        y = self.classifier(torch.cat([z_attn, hist], dim=0)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
