"""a171 — a158 adaptive temperature + learnable COVERAGE FLOOR (two diffuse knobs).

LIGHT, self-contained. Keeps a158 EXACTLY (global learnable temperature T = softplus(T_raw), which
CAN grow without bound) and adds ONE more adaptive knob that expresses the same pathology principle
from a different angle: a learnable uniform COVERAGE FLOOR. The grading prior is holistic, bag-wide
fibre density — so beyond letting attention spread via T, we let every patch keep a guaranteed
minimum share of the pooling weight (a floor of mean/uniform coverage):

    e_i   = W(V(h_i) * U(h_i))                 # gated attention energies
    T     = softplus(T_raw)                    # a158 temperature (can grow); init 1.0
    a_i   = softmax(e_i / T)                   # tempered attention
    eps   = sigmoid(eps_raw)                   # coverage-floor fraction in (0,1); init 0
    w_i   = (1 - eps) * a_i + eps * (1/N)      # convex mix with uniform -> guaranteed coverage
    z     = Σ w_i h_i ; y = classifier(z)

Difference from T alone: T->inf flattens toward uniform but DESTROYS all structure; the eps floor
keeps the softmax STRUCTURE while guaranteeing every patch a baseline weight (true diffuse coverage,
not collapse). T_raw init -> T=1 and eps_raw init -> eps=0 => exact a158/ABMIL baseline at
start; the optimiser earns each knob only if it generalises. Permutation- & size-invariant (uniform
= 1/N), deterministic at inference. 2 extra DOF over the baseline. No axis, no external file.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

_INV_SOFTPLUS_1 = math.log(math.expm1(1.0))  # T_raw init so softplus=1


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.T_raw = nn.Parameter(torch.tensor(_INV_SOFTPLUS_1))     # T = softplus -> 1.0 init
        self.eps_raw = nn.Parameter(torch.tensor(-6.0))              # sigmoid(-6) ~ 0.0025 ~ 0 init

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        n = e.shape[0]
        T = F.softplus(self.T_raw).clamp(min=1e-2)
        a = F.softmax(e / T, dim=0)
        eps = torch.sigmoid(self.eps_raw)
        w = (1.0 - eps) * a + eps * (1.0 / n)           # coverage floor (uniform mix)
        z = torch.mm(w.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
