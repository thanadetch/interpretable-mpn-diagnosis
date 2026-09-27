"""ctx_asgap — instance CONTEXTUALISATION before ASGAP pooling (base module for a374-a376).

Motivated by the published-baseline sweep run on 2026-08-08: TransMIL was the only published
method to reach ASGAP-level test QWK (virchow2 .9604 vs ASGAP .9588, titan mRec 87.6 vs 87.1),
despite a consistently weaker val score. TransMIL differs from ABMIL/ASGAP in exactly one place —
instances talk to each other through self-attention before being pooled — so the question this
module isolates is:

    does letting patches see each other help, once the pooling is held fixed at ASGAP's
    alpha-entmax gated attention?

MECHANISM

    h   = Bottleneck(x)                                   # [N, H], same as ASGAP
    h  <- h + Wo( SelfAttention(LN(h)) )                   # L contextualisation blocks
    h  <- h + FFN(LN(h))                                   # optional, per block
    e_i = W( tanh(V h_i) (.) sigmoid(U h_i) )              # ASGAP attention, unchanged
    a   = entmax_1.5(e) ;  z = sum_i a_i h_i ;  y = C(z)

`Wo` (the attention output projection) and the FFN output are ZERO-INITIALISED, so at step 0 the
model is bit-identical to ASGAP and any gain has to be learned rather than handed over by a
different initialisation.

WHY NO PPEG. TransMIL's position-encoding block squarifies the token sequence into an arbitrary
grid, which makes it order-dependent. This cohort has already been measured to carry no usable
signal in the real patch-grid coordinates, so an *arbitrary* grid is very unlikely to help and
would cost permutation invariance. `pos="none"` (the default) keeps the model permutation-
invariant; `pos="ppeg"` reintroduces TransMIL's block for a direct ablation of that claim.

READOUT
    "entmax"  ASGAP gated attention over the contextualised instances (the point of the module)
    "cls"     a class token, TransMIL-style, so the two readouts can be compared under identical
              contextualisation.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect

READOUTS = ("entmax", "cls")
POS = ("none", "ppeg")


class _Block(nn.Module):
    """Pre-norm self-attention + FFN, both residual paths zero-initialised."""

    def __init__(self, dim: int, heads: int, dropout: float, ffn: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.ffn = None
        if ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        if self.ffn is not None:
            x = x + self.ffn(self.norm2(x))
        return x


class _PPEG(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, N, D] — squarify, convolve, flatten back, trim to N
        N, D = x.shape[1], x.shape[2]
        S = int(math.ceil(math.sqrt(N)))
        pad = S * S - N
        z = torch.cat([x, x[:, :pad, :]], dim=1) if pad else x
        g = z.transpose(1, 2).reshape(1, D, S, S)
        g = g + self.proj(g) + self.proj1(g) + self.proj2(g)
        return g.flatten(2).transpose(1, 2)[:, :N, :]


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, layers: int = 1, heads: int = 4,
                 readout: str = "entmax", pos: str = "none", alpha: float = 1.5,
                 ffn: bool = True):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}")
        if pos not in POS:
            raise ValueError(f"pos must be one of {POS}")
        self.readout, self.pos, self.alpha = readout, pos, float(alpha)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.blocks = nn.ModuleList(
            [_Block(hidden_dim, heads, dropout, ffn) for _ in range(int(layers))])
        self.ppeg = _PPEG(hidden_dim) if pos == "ppeg" else None
        if readout == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        h = self.bottleneck(features).unsqueeze(0)              # [1, N, H]

        if self.readout == "cls":
            h = torch.cat([self.cls_token.to(h.dtype), h], dim=1)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if self.ppeg is not None and i == 0:
                if self.readout == "cls":
                    h = torch.cat([h[:, :1], self.ppeg(h[:, 1:])], dim=1)
                else:
                    h = self.ppeg(h)

        if self.readout == "cls":
            y = self.classifier(self.norm(h)[:, 0]).view(-1)
            n = features.shape[0]
            a = torch.full((n,), 1.0 / n, device=features.device) if return_attention else None
            return (y, a, None) if return_attention else (y, None, None)

        h = h.squeeze(0)                                        # [N, H]
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.alpha)
        y = self.classifier(torch.mv(h.t(), a)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, layers=1, readout="entmax", pos="none")
