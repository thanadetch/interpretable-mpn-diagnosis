"""a435 — ASGAP with a TWO-COVARIATE alpha residual: bag size AND magnification.

    alpha = clip( 1 + sigmoid(alpha_raw)  +  GAIN*(w_N * z_N  +  w_S * s * has_scale),  1.02, 2.0 )

    z_N        = (log N - 3.725) / 0.323          standardised bag size (a327 constants)
    s          = (log10(um/px) + 0.30) / 0.35     standardised magnification
    has_scale  = 1 when the scale bar was read, 0 otherwise (29% of ROIs)

Both coefficients start at 0, so at step 0 the model is a215/ASGAP bit for bit and can only
deviate if the gradient moves it; ROIs with no readable scale bar simply lose the second term and
keep the ordinary ASGAP alpha, so the unmeasurable group is never structurally special.

WHY THIS COMBINATION AND NOT ANOTHER
------------------------------------
Of every covariate tried on this task, bag size (a327) gave the largest mean gain over the fixed
baseline (+0.0092, winning 2 of 3 encoders) and magnification (a431) produced the only alpha gate
that demonstrably moves (w_s = -0.033/-0.047/-0.084, 17x its shuffled control). Neither has been
given to the same alpha at the same time; grep over 464 modules finds no file reading both.

PRE-TEST ON RECORD (frozen checkpoints, both coefficients grid-searched on val, applied to test):
val-fitted coefficients beat ASGAP on 2 of 3 encoders by +0.0025/+0.0023 and lose on TITAN by
-0.0026, while changing exactly 4 of 259 test predictions on every encoder, and the two-covariate
oracle ceiling (+0.0038/+0.0146/+0.0013) is no larger than the single-covariate ones. The
expectation recorded before running is therefore "indistinguishable from ASGAP".

CONTROL: `shuffle=True` redraws each ROI's magnification from the observed marginal, preserving the
alpha distribution and destroying only the ROI-to-scale correspondence. Bag size is left intact in
both arms, so the pair isolates the magnification term.
"""
from __future__ import annotations

from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

N_REF, N_SCALE = 3.725, 0.323
GAIN = 3.0
ALPHA_MIN, ALPHA_MAX = 1.02, 2.0


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
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))   # ASGAP global level -> 1.5
        self.w_n = nn.Parameter(torch.tensor(0.0))         # bag-size term, 0 => exactly ASGAP
        self.w_s = nn.Parameter(torch.tensor(0.0))         # magnification term
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)
        self._pool = np.array([v[0] for v in self.bank.table.values() if v[1] < 0.5], dtype=float)

    def _alpha(self, n_patches: int, s: float, u: float) -> torch.Tensor:
        z_n = (math.log(max(n_patches, 1)) - N_REF) / N_SCALE
        base = 1.0 + torch.sigmoid(self.alpha_raw)
        a = base + GAIN * self.w_n * z_n
        if u <= 0.5:                                        # scale readable
            if self.shuffle and len(self._pool):
                s = float(self._rng.choice(self._pool))
            a = a + GAIN * self.w_s * s
        return torch.clamp(a, ALPHA_MIN, ALPHA_MAX)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        out = {"alpha_base": round(float(1.0 + torch.sigmoid(self.alpha_raw)), 4),
               "w_n": round(float(self.w_n), 5), "w_s": round(float(self.w_s), 5)}
        for n in (13, 40, 112):
            out[f"N={n}"] = round(float(self._alpha(n, 0.0, 1.0)), 4)
        for v in (0.27, 1.06, 2.70):
            s = (np.log10(v) - LOG_REF) / LOG_SCALE
            out[f"{v:.2f}um"] = round(float(self._alpha(40, s, 0.0)), 4)
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self._alpha(features.shape[0], s, u))
        y = self.classifier(torch.mv(h.t(), a)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
