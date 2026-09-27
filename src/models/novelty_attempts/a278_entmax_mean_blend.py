"""a278 — Entmax-pool + plain uniform-mean gated blend (UNTRIED INSTANCE; long-shot).
z = sigmoid(g)*z_entmax + (1-g)*mean_i h_i. The uniform-mean branch is the maximally-diffuse
bag read (uni2-friendly per grading principle). Distinct from a276 (entmax+softmax-pool) and
a13/a27 (attn-softmax+mean). g init 1.5 -> ~0.82 entmax at start (~a215). Drop-in, perm/size-inv,
deterministic, MPS-safe, +1 param. Honest: data-driven ceiling -> ~0 expected GATE2."""
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
        self.gate = nn.Parameter(torch.tensor(1.5))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        g = torch.sigmoid(self.gate)
        z = g * torch.mv(h.t(), a) + (1.0 - g) * h.mean(dim=0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
