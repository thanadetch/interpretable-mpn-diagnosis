"""a277 — Entmax-pool + feature-wise max-pool gated blend (UNTRIED INSTANCE).

Blend the a215 1.5-entmax attention pool with a permutation/size-invariant feature-wise MAX over
patches (per-channel peak activation), via a learned scalar gate:
    z = sigmoid(g)*z_entmax + (1-sigmoid(g))*max_i h_i  (elementwise max over patches).
The max branch captures the strongest per-channel evidence anywhere in the bag (complementary to the
relevance-weighted mean); g lets the model mix. Feature-wise max-over-patches was not previously
blended with the entmax gated pool here. Drop-in, perm/size-invariant (max is both), deterministic,
MPS-safe, +1 param. Honest long-shot (data-driven ceiling).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


def entmax_bisect(z, alpha, n_iter=25):
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max() - (1.0 / am1) * 0.0
    lo = z.max() - 1.0
    tau_lo = lo; tau_hi = hi
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        if p.sum() > 1: tau_lo = tau
        else: tau_hi = tau
    p = torch.clamp(z - tau_hi, min=0) ** (1.0 / am1)
    return p / p.sum().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.gate = nn.Parameter(torch.tensor(1.0))   # sigmoid(1)=0.73 -> start mostly entmax (~a215)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        z_attn = torch.mv(h.t(), a)
        z_max = h.max(dim=0).values
        g = torch.sigmoid(self.gate)
        z = g * z_attn + (1.0 - g) * z_max
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
