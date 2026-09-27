"""a161 — Fibrosis-gated attention (LIGHT grading tweak of SimpleGatedMIL).

Keeps the simple gated attention exactly (the head that already works well) and makes ONE light,
grading-faithful change: the learned attention is MULTIPLICATIVELY re-weighted toward high-fibrosis
patches, then renormalised:

    a_i  = softmax(e_i)                         # the usual learned gated attention
    m_i  = σ(κ · ẑ_i)                           # fibrosis membership; ẑ = within-bag std fibrosis-axis score
    w_i  = a_i·m_i / Σ_j a_j·m_j                # fibrosis-gated, renormalised attention
    z    = Σ w_i·h_i ;  y = classifier(z)

v = train G3−G0 mean-difference axis (IN-MEMORY via a155._train_axis, leakage-safe, no file; fixed
buffer). κ is learnable and init = 1.0 (ACTIVE — the grading gate is on from the start, so unlike an
init-at-zero bias it does not trivially collapse to the baseline). Grading story: the attention the
model learns is focused onto the diffuse fibrotic signal. 1 extra DOF (κ) over the baseline.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _train_axis, _current_seed


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True):
        super().__init__()
        self.feat_dim = input_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.kappa = nn.Parameter(torch.tensor(1.0))  # ACTIVE fibrosis-gating from init
        v = None
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a161] unknown input_dim {input_dim} (no fibrosis-axis mapping).")
            v = got[0].float().view(-1).clone()
        if v is None:
            v = F.normalize(torch.randn(input_dim), dim=0)
        self.register_buffer("v", v)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        s = feats @ self.v
        if s.shape[0] >= 2:
            s = (s - s.mean()) / s.std().clamp(min=1e-6)
        else:
            s = torch.zeros_like(s)
        m = torch.sigmoid(self.kappa * s)
        w = a * m
        w = w / w.sum().clamp(min=1e-8)
        z = torch.mv(h.t(), w)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, w, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
