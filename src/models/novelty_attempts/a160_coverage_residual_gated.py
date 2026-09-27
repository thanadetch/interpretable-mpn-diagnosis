"""a160 — Coverage-residual gated attention (diffuse-density correction on the strong head, init = baseline).

LIGHT, self-contained. Reads the grade as the strong ABMIL output PLUS a small additive
correction proportional to the bag-wide fibrosis OCCUPANCY (the diffuse-density grading prior):

    y0  = ABMIL readout                                  # the head that already works
    cov = mean_i σ(α·(⟨feat_i, v⟩ − τ))                           # soft fraction of fibrotic patches
    y   = y0 + δ·(cov − 0.5)                                       # δ init 0 ⇒ EXACTLY the baseline

v = train G3−G0 axis (IN-MEMORY via a155._train_axis, no file); α,τ initialised from the per-grade
train anchors. δ init 0 means the model starts at the baseline readout and only adds the holistic
coverage correction if it improves val — so it cannot degrade titan's near-ceiling head at init.
Low DOF (δ, α, τ; v fixed). Grading-faithful: an explicit "overall fibre-meshwork density" term.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a155_axis_attention_density import _train_axis, _current_seed


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True):
        super().__init__()
        assert num_classes == 1
        self.feat_dim = input_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.delta = nn.Parameter(torch.tensor(0.0))  # init 0 = exact baseline readout
        tau_init, alpha_init, v = 0.0, 1.0, None
        if warm_start:
            got = _train_axis(input_dim, _current_seed())
            if got is None:
                raise ValueError(f"[a160] unknown input_dim {input_dim} (no fibrosis-axis mapping).")
            axis, anchors = got
            v = axis.float().view(-1).clone()
            amin, amax = float(anchors.min()), float(anchors.max())
            span = max(amax - amin, 1e-6)
            tau_init = 0.5 * (amin + amax)
            alpha_init = 4.0 / span
        if v is None:
            v = F.normalize(torch.randn(input_dim), dim=0)
        self.register_buffer("v", v)  # FIXED axis
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.alpha_raw = nn.Parameter(torch.tensor(math.log(math.expm1(max(float(alpha_init), 1e-3)))))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        attn = F.softmax(e, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y0 = self.classifier(z).view(-1)
        s = feats @ self.v
        cov = torch.sigmoid(F.softplus(self.alpha_raw) * (s - self.tau)).mean()
        y = y0 + self.delta * (cov - 0.5)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, warm_start=True)
