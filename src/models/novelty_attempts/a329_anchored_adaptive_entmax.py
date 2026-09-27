"""a329 — Anchored Adaptive-Alpha ASGAP (AA-ASGAP).

Design lesson chain:
  a215  : global scalar alpha -> INERT (stuck ~1.5), good but not learnable.
  a327  : size-scalar alpha    -> still inert (scalars get too little gradient).
  a328  : content-head alpha   -> genuinely LEARNABLE (per-bag std ~0.2) BUT it drifts sparse
                                  (TITAN mean 1.67) and test drops -> free adaptivity leaves the
                                  proven-optimal region and HURTS.

a329 keeps alpha learnable + per-bag adaptive (content head, as a328) but **anchors it at the
empirically-optimal 1.5 and bounds the deviation to a narrow band**, so training can only make
small, safe per-bag adjustments around the sweet spot and can never drift into the harmful
over-sparse region that sank a328:

    alpha = 1.5 + DELTA * tanh( head(desc) )      # alpha in (1.5-DELTA, 1.5+DELTA), DELTA=0.15

desc = [ mean_n h , std_n h , z_N ]  (content + standardised log bag size, small-data prior).
Head is zero-initialised -> alpha starts at exactly 1.5 for every bag (== a215/a268), then learns
bounded per-bag deviations. tanh saturates gracefully so it cannot exceed the band. This is the
"what kind of learnable" answer: learnable **anchored at the optimum with a bounded band**, not
free-range. Permutation/size-invariant, deterministic eval, concept-free. ~258 params over a215.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

LOG_REF = 3.725
LOG_SCALE = 0.323
DELTA = 0.15   # half-width of the alpha band around 1.5 -> alpha in [1.35, 1.65]


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
        self.alpha_head = nn.Linear(2 * hidden_dim + 1, 1)
        nn.init.zeros_(self.alpha_head.weight)   # start at alpha=1.5 for every bag
        nn.init.zeros_(self.alpha_head.bias)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._last_alpha: Optional[float] = None

    def _alpha(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        z_n = torch.as_tensor((math.log(max(n, 1)) - LOG_REF) / LOG_SCALE,
                              dtype=h.dtype, device=h.device).view(1)
        desc = torch.cat([h.mean(0), h.std(0, unbiased=False), z_n], dim=0)
        a = 1.5 + DELTA * torch.tanh(self.alpha_head(desc).squeeze(-1))  # in (1.5-DELTA, 1.5+DELTA)
        self._last_alpha = float(a.detach())
        return a

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        alpha = self._alpha(h)
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
