"""scale_formula_alpha — alpha is a CONTINUOUS FORMULA of the ROI's measured um/px, and the
ROIs whose scale bar cannot be read are PINNED at 1.5.

    s      = (log10(um/px) - LOG_REF) / LOG_SCALE      standardised scale
    alpha  = 1 + sigmoid( GAIN * (w_s * s + b) )       measurable ROIs, one value in (1, 2)
    alpha  = 1.5                                       unmeasurable ROIs, fixed, no gradient

Two learnable scalars (`w_s`, `b`), both initialised to zero, so every ROI starts at exactly
alpha = 1.5 — identical to ASGAP — and any spread afterwards is something the scale produced.
GAIN amplifies the pre-sigmoid term; without it a bare scalar gets too small a gradient on 30
training patients, the failure a215's learnable alpha ran into and the reason a327 introduced it.

WHY PIN THE UNKNOWNS — this is the point of the module
------------------------------------------------------
47% of ROIs (624 of 1330) have a scale bar the detector cannot read. Every earlier attempt gave
that group its own free parameter: `scale_gated_alpha` (a409) feeds an `unknown` flag into the
gate, `scale_band_alpha` (a413) and `scale_pick_alpha` (a415) each keep a fourth band for it. The
leak-free lookup analysis then showed that essentially ALL of the measured benefit came from that
fourth group and none from the measured-scale bins — which is not a magnification effect at all,
since "the scale bar is unreadable" is a property of image quality and cropping.

Pinning the unknowns at 1.5 removes that channel entirely. Whatever this model gains or loses is
attributable to the measured physical scale of the other 53%, and nothing else.

Compared with the earlier three: a409 mixes five fixed orders, so its effective alpha is an
average and is dragged toward the centre by construction (it only used 1.478-1.521); a413 frees
the range per band but bins the scale; a415 commits to one of five per band and picked 1.50 in
53 of 60 cells. This one is continuous in BOTH the input and the output, which is the form the
hypothesis is actually stated in: sparser pooling at high magnification because one patch then
covers only 60-100 um and the fibrotic focus sits in a few patches, flatter pooling at low
magnification because one patch already covers 160-600 um and the signal is spread.

CONTROL
-------
`shuffle=True` permutes which measured scale each ROI is assigned, keeping the marginal
distribution of scales and of alphas and destroying only the ROI-to-scale correspondence. The
unknown group stays pinned in both arms, so the two differ in exactly one thing.

`alpha_report()` returns the learned alpha across the cohort's real magnifications, so the mapping
can be read directly instead of inferred from accuracy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

GAIN = 3.0
PINNED_ALPHA = 1.5          # the unmeasurable ROIs never move off ASGAP's order
UM_GRID = (0.27, 0.36, 0.41, 0.54, 0.82, 1.06, 1.35, 2.70)


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0):
        super().__init__()
        self.shuffle, self.seed = bool(shuffle), int(seed)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.w_s = nn.Parameter(torch.zeros(()))     # scale coefficient
        self.b = nn.Parameter(torch.zeros(()))       # sigmoid(0)=0.5 -> alpha=1.5 at init
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)
        # the shuffled arm needs a pool of real scales to draw from; collect them once
        self._pool = np.array([v[0] for v in self.bank.table.values() if v[1] < 0.5], dtype=float)

    def _alpha(self, s: float, u: float):
        if u > 0.5:                                   # scale bar unreadable -> pinned, no gradient
            return torch.tensor(PINNED_ALPHA, device=self.w_s.device, dtype=self.w_s.dtype)
        if self.shuffle and len(self._pool):
            s = float(self._rng.choice(self._pool))   # keep the marginal, break the correspondence
        return 1.0 + torch.sigmoid(GAIN * (self.w_s * s + self.b))

    @torch.no_grad()
    def alpha_report(self) -> dict:
        out = {}
        for v in UM_GRID:
            s = (np.log10(v) - LOG_REF) / LOG_SCALE
            out[f"{v:.2f}"] = round(float(1.0 + torch.sigmoid(GAIN * (self.w_s * s + self.b))), 4)
        out["unknown"] = PINNED_ALPHA
        out["w_s"] = round(float(self.w_s), 5)
        out["b"] = round(float(self.b), 5)
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        alpha = self._alpha(s, u)
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, alpha)
        y = self.classifier(torch.mv(h.t(), a)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
