"""a207 - Perceiver-style latent pooling. NEW: M learned latents cross-attend to patches, pool latents.

A small set of M learned latent vectors cross-attend to the (bottlenecked) patches (Perceiver/Set-
Transformer PMA), are refined by self-attention among themselves, then averaged into the bag rep.
This bottlenecks the bag through a fixed-size learned latent array -- a different inductive bias from
per-patch gated attention. Uses torch MultiheadAttention (no new dep). Permutation/size-invariant,
deterministic at eval. Concept-free, self-contained.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, M=8, n_heads=4):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.latents = nn.Parameter(torch.randn(M, hidden_dim) * (hidden_dim ** -0.5))
        self.cross = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features).unsqueeze(0)              # [1,N,D]
        lat = self.latents.unsqueeze(0)                         # [1,M,D]
        c, _ = self.cross(lat, h, h, need_weights=False)
        lat = self.norm1(lat + c)
        s, _ = self.self_attn(lat, lat, lat, need_weights=False)
        lat = self.norm2(lat + s)
        z = lat.squeeze(0).mean(dim=0)                          # pool latents
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
