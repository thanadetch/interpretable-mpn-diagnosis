"""a373 — TransMIL (Shao et al., NeurIPS 2021), "TransMIL: Transformer based Correlated
Multiple Instance Learning for Whole Slide Image Classification".

Written from the paper and its reference implementation (github.com/szc19990412/TransMIL),
adapted to this project's ordinal-regression head. The third standard published MIL baseline the
codebase has never run on the grading task.

MECHANISM
  1. project instances to 512-d, prepend a learnable CLASS TOKEN
  2. TPT layer 1  — self-attention over the token sequence
  3. PPEG         — Pyramid Position Encoding Generator: the instance tokens are squarified into
                    an HxW grid (padded by repeating the head of the sequence, exactly as the
                    reference does), passed through three depthwise convolutions (7x7, 5x5, 3x3)
                    whose outputs are ADDED back as a residual, then flattened. This is what gives
                    TransMIL a notion of "correlation" between instances without real coordinates.
  4. TPT layer 2  — self-attention again
  5. LayerNorm on the class token -> linear head

DEVIATIONS (honest, and both are forced by bag size)
  - FULL SELF-ATTENTION INSTEAD OF NYSTROM. The paper uses Nystrom attention with 256 landmarks
    to make O(N^2) tractable for N in the thousands. ROI bags here hold ~44 instances (max 112),
    far fewer than 256 landmarks, so Nystrom's landmark approximation would reduce to full
    attention anyway. Full attention is therefore used and is exact, not an approximation of the
    paper.
  - PPEG's square padding is a much larger fraction of the sequence at N=44 (grid 7x7 = 49 slots)
    than at N=4000. That is a genuine property of transplanting TransMIL to ROI scale, not an
    implementation shortcut, and is reported as such.
  - REGRESSION HEAD: num_classes=1 with this project's ordinal target, not the paper's
    cross-entropy.

Deterministic at eval. NOT permutation-invariant — PPEG imposes an (arbitrary) sequence order,
which is exactly the property this cohort has already been shown not to benefit from (patch-grid
coordinates carry no usable signal here), so a loss against ABMIL would be informative in itself.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class _PPEG(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: [1, 1+H*W, D] — token 0 is the class token
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(1, -1, H, W)
        out = cnn_feat + self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        out = out.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), out), dim=1)


class _TransLayer(nn.Module):
    """Pre-norm self-attention block (the paper's TPT layer)."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.attn(h, h, h, need_weights=False)[0]


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1,
                 hidden_dim: int = 512, dropout: float = 0.0, heads: int = 8,
                 use_ppeg: bool = True):
        super().__init__()
        self.dim = hidden_dim
        self.use_ppeg = use_ppeg
        self._fc1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.layer1 = _TransLayer(hidden_dim, heads)
        self.pos_layer = _PPEG(hidden_dim) if use_ppeg else None
        self.layer2 = _TransLayer(hidden_dim, heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 2:
            features = features.unsqueeze(0)                  # [1, N, D]
        h = self._fc1(features)                               # [1, N, dim]

        # squarify: pad to H*W by repeating the head of the sequence (the reference's scheme)
        N = h.shape[1]
        H = W = int(math.ceil(math.sqrt(N)))
        pad = H * W - N
        if pad > 0:
            h = torch.cat([h, h[:, :pad, :]], dim=1)

        h = torch.cat((self.cls_token.expand(1, -1, -1).to(h.dtype), h), dim=1)
        h = self.layer1(h)
        if self.pos_layer is not None:
            h = self.pos_layer(h, H, W)
        h = self.layer2(h)

        y = self._fc2(self.norm(h)[:, 0]).view(-1)
        if return_attention:
            # uniform placeholder: the class token's attention is not exposed by nn.MultiheadAttention here
            a = torch.full((features.shape[1],), 1.0 / features.shape[1], device=features.device)
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
