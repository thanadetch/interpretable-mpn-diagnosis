"""a175 — Diffuse-temperature (a158) + embedding-noise (a172) composition (data-driven combo).

LIGHT, self-contained. The two best-behaved candidates each helped a DIFFERENT backbone's test:
a158 (learnable diffuse temperature T) raised virchow2 test (+0.0073, all grades up) and a172
(train-time embedding noise) raised titan test (+0.0033). They are mechanistically orthogonal
(T reshapes attention; noise regularises the embedding), so this composes BOTH on the same
ABMIL head to test whether the gains stack across backbones (the only honest path to GATE2,
which needs ALL backbones up at once):

    h_i   = bottleneck(x_i)
    h_i   = h_i + sigma * eps_i          (TRAIN ONLY; a172 denoising; off at inference)
    e_i   = W(V(h) * U(h))
    T     = softplus(T_raw)              (a158 diffuse temperature; init 1.0)
    a     = softmax(e / T) ; z = Σ a_i h_i ; y = classifier(z)

T_raw init -> T=1.0 and sigma init 0.1; at inference (noise off) with T~1 it reduces to the baseline,
so each knob is earned, not imposed. DETERMINISTIC at inference. Permutation/size-invariant. 2 extra
DOF over the baseline. No axis, no external file.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_INV_SOFTPLUS_1 = math.log(math.expm1(1.0))


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.T_raw = nn.Parameter(torch.tensor(_INV_SOFTPLUS_1))     # a158 T=1.0 init
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.1)))   # a172 noise std

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        if self.training:
            h = h + self.log_sigma.exp() * torch.randn_like(h)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        T = F.softplus(self.T_raw).clamp(min=1e-2)
        attn = F.softmax(e / T, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
