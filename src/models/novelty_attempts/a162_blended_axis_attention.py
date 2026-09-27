"""a162 — Blended fibrosis-axis attention (LIGHT grading tweak of SimpleGatedMIL).

Keeps the simple gated attention and BLENDS it with an explicit fibrosis-axis attention head:

    a_gate = softmax(e)                         # the learned gated attention (works well)
    a_axis = softmax(λ · ẑ)                      # fibrosis-density attention (ẑ = within-bag std axis score)
    w      = (1−β)·a_gate + β·a_axis             # convex blend, β = σ(β_raw) init 0.5 (ACTIVE)
    z      = Σ w_i·h_i ;  y = classifier(z)

v = train G3−G0 axis (IN-MEMORY via a155._train_axis, leakage-safe, no file; fixed buffer). β init
0.5 and λ init 1.0 → the fibrosis-density attention is ON from the start (does not collapse to the
baseline at init). Grading story: the bag read combines the learned attention with an explicit
diffuse-fibrosis-density attention — a literal "weight patches by how fibrotic they are, blended
with what the model learns." Light: 2 extra DOF (β, λ) over the baseline gated head.
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
        self.beta_raw = nn.Parameter(torch.tensor(0.0))  # β = σ(β_raw) = 0.5 init (active blend)
        self.lam = nn.Parameter(torch.tensor(1.0))       # axis-attention sharpness, active
        v = None
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a162] unknown input_dim {input_dim} (no fibrosis-axis mapping).")
            v = got[0].float().view(-1).clone()
        if v is None:
            v = F.normalize(torch.randn(input_dim), dim=0)
        self.register_buffer("v", v)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a_gate = F.softmax(e, dim=0)
        s = feats @ self.v
        if s.shape[0] >= 2:
            s = (s - s.mean()) / s.std().clamp(min=1e-6)
        else:
            s = torch.zeros_like(s)
        a_axis = F.softmax(self.lam * s, dim=0)
        beta = torch.sigmoid(self.beta_raw)
        w = (1.0 - beta) * a_gate + beta * a_axis
        z = torch.mv(h.t(), w)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, w, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
