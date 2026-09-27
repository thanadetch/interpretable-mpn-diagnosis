"""a276 — Entmax(sparse) + Softmax(dense) gated blend (UNTRIED INSTANCE).

Pool the SAME a215 gated scores TWO ways — 1.5-entmax (sparse, a215) and softmax (dense, =alpha-1
diffuse) — and blend the two pooled vectors with a single learned scalar gate g:
    z = sigmoid(g)*z_entmax + (1-sigmoid(g))*z_softmax.
The dense/softmax branch keeps the diffuse bag-wide density (uni2-friendly per the grading principle)
while the entmax branch keeps the sparse-focal read; g lets the model pick the mix from data.
Distinct from a272 (dual-alpha CONCAT of 1.2&1.5) and a27/a13 (attn vs MEAN-pool blend): here it is a
learned BLEND of entmax-pool vs softmax-pool of the same gated scores. g init 0 -> 50/50.
Drop-in, perm/size-invariant, deterministic, MPS-safe, +1 param.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gate = nn.Parameter(torch.zeros(1))   # blend logit; sigmoid(0)=0.5
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a_sparse = entmax_bisect(e, self.ALPHA)
        a_dense = F.softmax(e, dim=0)
        g = torch.sigmoid(self.gate)
        z = g * torch.mv(h.t(), a_sparse) + (1.0 - g) * torch.mv(h.t(), a_dense)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a_sparse, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
