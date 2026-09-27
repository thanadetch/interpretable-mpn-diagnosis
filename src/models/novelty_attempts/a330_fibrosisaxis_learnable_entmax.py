"""a330 — Fibrosis-Axis-Conditioned Learnable-Alpha ASGAP (FA-ASGAP).

The learnable-alpha attempts a327/a328/a329 all failed because the alpha's conditioning signal was
either absent (bare scalar -> inert) or grade-uninformative (content mean/std, bag size -> alpha
moves but toward a worse region). Here the alpha is conditioned on a *grade-relevant* signal: the
bag's projection onto the **data-derived fibrosis axis** (v = train-derived direction separating
low- vs high-grade, precomputed from real grade labels; permitted per the no-zeroshot-labels rule).

Hypothesis: high-fibrosis bags (large axis projection, G2/G3 — diffuse dense reticulin) want DENSER
pooling (lower alpha -> aggregate the many fibrotic patches); low-fibrosis bags (G0/G1 — focal) want
SPARSER pooling (higher alpha -> focus). A tiny learnable affine map (w, b) turns the projection into
alpha; direction is learned, not hard-coded.

    proj  = mean_n(features) . v_unit                        # scalar, grade-correlated
    alpha = 1 + sigmoid( GAIN * (w * proj_std + b) )         # in (1,2), learnable w,b

Only +2 scalar params over a215. virchow2 only (axis precomputed for virchow2, seed-2 train split);
falls back to fixed 1.5 if the axis is unavailable / dim mismatch. Permutation/size-invariant,
deterministic eval, uses no test-time labels.
"""
from __future__ import annotations
from typing import Optional, Tuple
import os
import torch
import torch.nn as nn

_AXIS_PATH = "data/prototypes_virchow2_reti_train_seed2.pt"
PROJ_CENTER = -1.3   # ~mean of per-grade axis projections (R_mean: -11,-4,1.4,8.5)
PROJ_SCALE = 7.4     # ~std of per-grade axis projections
GAIN = 2.0


def entmax_bisect(z, alpha, n_iter=25):
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max()
    lo = z.max() - 1.0
    tau_lo = lo; tau_hi = hi
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        Z = p.sum()
        if Z > 1: tau_lo = tau
        else: tau_hi = tau
    p = torch.clamp(z - tau_hi, min=0) ** (1.0 / am1)
    return p / p.sum().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.w = nn.Parameter(torch.tensor(0.0))   # learnable slope (alpha vs fibrosis proj)
        self.b = nn.Parameter(torch.tensor(0.0))   # learnable offset -> alpha 1.5 at proj center
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # load fibrosis axis (grade-derived direction) as a fixed unit-vector buffer
        axis = None
        if os.path.exists(_AXIS_PATH):
            d = torch.load(_AXIS_PATH, map_location="cpu", weights_only=False)
            a = d.get("axis")
            if a is not None and a.numel() == input_dim:
                axis = (a / a.norm().clamp(min=1e-8)).float()
        self.register_buffer("axis", axis if axis is not None else torch.zeros(input_dim))
        self._has_axis = axis is not None
        self._last_alpha: Optional[float] = None

    def _alpha(self, features: torch.Tensor) -> torch.Tensor:
        if not self._has_axis:
            self._last_alpha = 1.5
            return torch.tensor(1.5, dtype=features.dtype, device=features.device)
        proj = torch.dot(features.mean(0), self.axis.to(features.dtype))
        proj_std = (proj - PROJ_CENTER) / PROJ_SCALE
        a = 1.0 + torch.sigmoid(GAIN * (self.w * proj_std + self.b))
        self._last_alpha = float(a.detach())
        return a

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        alpha = self._alpha(features)             # grade-informed, learnable
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
