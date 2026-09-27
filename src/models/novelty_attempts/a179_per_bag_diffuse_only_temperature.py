"""a179 — Per-bag diffuse-only temperature (a170 fixed: per-bag T but constrained T>=1).

a170 made the temperature depend on each bag's attention entropy, but left it FREE -> on virchow2 it
drove some bags to T<1 (sharper) and crashed test (-0.0345). The diagnosis (ad-hoc sweep) showed
sharpening hurts every backbone. So a179 keeps the per-bag adaptivity but constrains it to the safe
diffuse-only region:

    H     = normalised attention entropy of softmax(e) in [0,1]   (bag descriptor)
    T_bag = 1 + softplus(g0 + g1 * (H - 0.5))    (>= 1 for EVERY bag: flatten-or-neutral only)
    a     = softmax(e / T_bag) ; z = Σ a_i h_i ; y = classifier(z)

g0 init -> softplus~0 -> T~1; g1 init 0. So init = baseline, and no bag can ever be sharpened below
the standard softmax. Self-contained, 2 DOF, deterministic at inference, permutation/size-invariant.
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
        self.g0 = nn.Parameter(torch.tensor(-6.0))   # softplus(g0)~0 -> T~1
        self.g1 = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        n = e.shape[0]
        if n >= 2:
            a0 = F.softmax(e, dim=0)
            H = (-(a0 * a0.clamp(min=1e-12).log()).sum()) / math.log(n)
        else:
            H = e.new_tensor(0.5)
        T = 1.0 + F.softplus(self.g0 + self.g1 * (H - 0.5))   # >= 1 per bag
        attn = F.softmax(e / T, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
