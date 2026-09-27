"""a317 — Concentration-gated SLERP pool: direction-only correction, self-gated by vMF concentration.

GENUINELY NEW (no slerp/vMF-resultant pooling gate on disk). Like a316, leaves attention weights AND
pooled magnitude bit-for-bit baseline; acts ONLY on the pooled DIRECTION, via a spherical-linear
interpolation between the Euclidean mean direction and the spherical (mean-resultant) direction, with
the interpolation amount g = 1 - Rbar GATED by the von Mises-Fisher mean resultant length Rbar in [0,1].
DOUBLE uni2-safety guarantee: on concentrated (uni2) bags both the inter-direction angle omega->0 AND
the gate g->0, each independently forcing z = R * dir_euclid = baseline softmax-mean.

DISCLOSURE: exploratory fish (user's "just research, keep searching"); data ceiling may cap it; we run,
disclose, multi-seed-audit any pass. ZERO extra params (197,250). Honest caveat: a parameter-free gate
may be too weak/strong on the 214-ROI cohort; it degrades gracefully to baseline as Rbar->1. Self-
contained, concept-free, permutation/size-invariant, deterministic eval, MPS-safe.
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
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        c = torch.mv(h.t(), a)
        R = c.norm()
        u = h / (h.norm(dim=1, keepdim=True) + 1e-6)
        Rvec = torch.mv(u.t(), a)                                                       # [H]
        Rbar = Rvec.norm()                                                              # in [0,1]
        dir_euclid = c / (c.norm() + 1e-6)
        dir_sphere = Rvec / (Rbar + 1e-6)
        g = 1.0 - Rbar                                                                  # spread gate (param-free)
        omega = (dir_euclid @ dir_sphere).clamp(-1 + 1e-6, 1 - 1e-6).arccos()
        if omega < 1e-4:
            direction = dir_euclid
        else:
            so = torch.sin(omega)
            direction = (torch.sin((1.0 - g) * omega) / so) * dir_euclid + (torch.sin(g * omega) / so) * dir_sphere
        z = R * direction
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
