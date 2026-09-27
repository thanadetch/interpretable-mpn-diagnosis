"""a318 — Rao-Blackwell patch-dropout pool: analytic de-concentration from the attention SHAPE.

GENUINELY NEW (no delta-method / keep_prob / Rao-Blackwell pooling on disk). The analytic
marginalization of patch-dropout: a deterministic de-concentration term derived from the attention
SHAPE (kappa = sum_i a_i^2 = collision prob = 1/n_eff), NOT from magnitude (a248 James-Stein) and NOT a
learnable scalar (a312, inert). It pulls the pool away from its own MOST-concentrated direction (the
self-weighted pool z2) in proportion to concentration kappa — a different remedy than shrink-toward-mean.

z_RB = z_attn + ((1-rho)/rho) * kappa * (z_attn - z2)

VANISHES EXACTLY on a one-hot bag (z_attn = z2 = h_j), so uni2-safe by construction; correction bounded
by (1-rho)/rho and shrinks as the pool sharpens. rho is a fixed BUFFER (0.85), not learnable -> cannot
go inert. DISCLOSURE: exploratory fish (user's "just research"); data ceiling may cap it; run, disclose,
multi-seed-audit any pass. ZERO extra params (197,250; rho is a buffer). Self-contained, concept-free,
permutation/size-invariant, deterministic eval, MPS-safe.
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
        self.register_buffer("rho", torch.tensor(0.85))                                # fixed keep-prob (not learned)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        z_attn = torch.mv(h.t(), a)                                                     # [H] baseline pool
        kappa = (a * a).sum()                                                           # collision prob = 1/n_eff
        a2 = a * a
        z2 = torch.mv(h.t(), a2) / (a2.sum() + 1e-12)                                   # [H] self-weighted (sharper) pool
        z_RB = z_attn + ((1.0 - self.rho) / self.rho) * kappa * (z_attn - z2)
        y = self.classifier(z_RB).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
