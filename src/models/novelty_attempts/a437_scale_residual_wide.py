"""a437 — Scale-Residual ASGAP with a WIDE gate (GAIN=10). Scale only; no bag-size term.

The user's design: keep ASGAP exactly as it is -- one global LEARNABLE entmax order that converges
to ~1.50 -- and add a small scale-dependent RESIDUAL on top of it:

    alpha(ROI) = clip( 1 + sigmoid(alpha_raw)  +  GAIN * w_s * s * has_scale ,  1.02, 2.0 )

    s          = (log10(um/px) - LOG_REF) / LOG_SCALE     standardised magnification
    has_scale  = 1 when the scale bar was read, 0 otherwise

Two properties distinguish this from every earlier scale model in the repo:

1. **Strict generalisation.**  ``w_s`` is initialised at 0, so at step 0 the model IS a215/ASGAP,
   bit for bit, and it can only deviate if the gradient pushes it.  a417/a419/a421/a423 instead
   replace ASGAP's global level with their own ``b`` term.

2. **No hard pin.**  ROIs whose scale bar is unreadable (29%) simply get ``has_scale = 0`` and
   therefore the ordinary ASGAP alpha -- not a constant 1.5 detached from the graph.  The pinned
   family (a417-a429) made the unmeasurable group structurally different from the rest, which is
   what let the `unknown` bin act as a patient-group indicator.

GAIN amplifies only the residual's gradient; it does not change the w_s = 0 equivalence.

PRE-TEST (frozen ASGAP checkpoint, alpha = alpha_ASGAP + w*s, w fitted on val): changing w moved
just 1 / 5 / 1 of 259 test ROIs on virchow2 / uni2 / titan even when the induced alpha span was
0.91, and the TEST-fitted oracle chose w = 0.00 exactly on titan.  So the expectation on record
before running is "indistinguishable from ASGAP".

CONTROL: `shuffle=True` gives each measurable ROI a scale drawn from the observed marginal instead
of its own, preserving the alpha distribution and destroying only the ROI-to-scale correspondence.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

GAIN = 10.0   # a431 used 3.0; raised so the learned w_s can span the range the oracle wants
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
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))   # ASGAP's global level -> alpha 1.5
        self.w_s = nn.Parameter(torch.tensor(0.0))         # scale residual, 0 => exactly ASGAP
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)
        self._pool = np.array([v[0] for v in self.bank.table.values() if v[1] < 0.5], dtype=float)

    def _alpha(self, s: float, u: float) -> torch.Tensor:
        base = 1.0 + torch.sigmoid(self.alpha_raw)
        if u > 0.5:                                        # scale unreadable -> plain ASGAP alpha
            return torch.clamp(base, ALPHA_MIN, ALPHA_MAX)
        if self.shuffle and len(self._pool):
            s = float(self._rng.choice(self._pool))
        return torch.clamp(base + GAIN * self.w_s * s, ALPHA_MIN, ALPHA_MAX)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        out = {}
        for v in (0.27, 0.36, 0.54, 1.06, 2.70):
            s = (np.log10(v) - LOG_REF) / LOG_SCALE
            out[f"{v:.2f}"] = round(float(self._alpha(s, 0.0)), 4)
        out["unknown"] = round(float(self._alpha(0.0, 1.0)), 4)
        out["w_s"] = round(float(self.w_s), 5)
        out["alpha_base"] = round(float(1.0 + torch.sigmoid(self.alpha_raw)), 4)
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self._alpha(s, u))
        y = self.classifier(torch.mv(h.t(), a)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
