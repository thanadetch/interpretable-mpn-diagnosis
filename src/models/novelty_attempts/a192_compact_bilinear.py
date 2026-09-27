"""a192 - Compact bilinear pooling (count-sketch). NEW family: full-dim second-order via sketch.

a180 took a small-dim covariance; a192 captures the full second-order outer-product h h^T COMPACTLY
via Count-Sketch + element-wise product in sketch space (Gao et al. 2016, "Compact Bilinear Pooling"),
so the bag encodes second-order feature interactions across the whole hidden space without the
d^2 blow-up. Attention-weighted. Concept-free, self-contained, permutation/size-invariant, det. at eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, sketch=256):
        super().__init__()
        self.S = sketch
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        # fixed count-sketch params (buffers, not learned)
        g = torch.Generator().manual_seed(0)
        self.register_buffer("h_idx", torch.randint(0, sketch, (hidden_dim,), generator=g))
        self.register_buffer("s_sgn", (torch.randint(0, 2, (hidden_dim,), generator=g).float() * 2 - 1))
        self.classifier = nn.Linear(sketch, num_classes)

    def _sketch(self, x):  # x: [N,hidden] -> [N,S]
        out = x.new_zeros(x.shape[0], self.S)
        out.index_add_(1, self.h_idx, x * self.s_sgn.unsqueeze(0))
        return out

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        sk = self._sketch(h)                                 # [N,S]
        fft = torch.fft.rfft(sk, dim=1)
        cbp = torch.fft.irfft(fft * fft, n=self.S, dim=1)    # compact bilinear per patch (h outer h)
        z = torch.mv(cbp.t(), a)                             # attention-weighted second-order
        z = torch.sign(z) * torch.sqrt(z.abs() + 1e-8)       # signed sqrt norm
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
