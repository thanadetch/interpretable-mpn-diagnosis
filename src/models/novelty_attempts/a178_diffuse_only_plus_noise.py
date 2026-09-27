"""a178 — Diffuse-only temperature (T>=1) + embedding noise (holistic-safe + regularised).

Composition of the two mechanisms that EACH helped a different backbone, both constrained safe:
  - a176 diffuse-only temperature (T>=1): helped uni2 by forbidding the harmful over-sharpen.
  - a172 train-time embedding noise: gave the largest titan test gain (regularisation).
Hypothesis: a176 protects the weak backbones (no sharpening) while a172's denoising lifts the strong
ones -> the combination might raise test on ALL backbones at once (the only path to GATE2).

    h = bottleneck(x); if train: h += sigma*eps         (a172)
    T = 1 + softplus(.)  (>=1, a176) ; a = softmax(e/T) ; z = Σ a_i h_i ; y = classifier(z)

init T~1, sigma=0.1; at inference (noise off, T~1) -> baseline. Self-contained, 2 DOF, deterministic
at inference, permutation/size-invariant.
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
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.T_excess_raw = nn.Parameter(torch.tensor(-6.0))         # T = 1 + softplus -> ~1.0
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.1)))   # noise std

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        if self.training:
            h = h + self.log_sigma.exp() * torch.randn_like(h)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        T = 1.0 + F.softplus(self.T_excess_raw)
        attn = F.softmax(e / T, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
