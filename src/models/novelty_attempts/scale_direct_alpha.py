"""scale_direct_alpha — learn ONE entmax order per ROI as a function of its measured um/px.

Unlike `scale_gated_alpha.py`, which blends five fixed orders and is therefore pulled toward the
middle by construction (its learned effective alpha only spanned 1.478-1.518 although 1.0-2.0 was
available), this parameterises alpha directly:

    s      = (log10(um/px) - LOG_REF) / LOG_SCALE      standardised scale, 0 when unmeasurable
    u      = 1 if the scale bar could not be read, else 0
    alpha  = 1 + sigmoid( w_s * s + w_u * u + b )      one value in (1, 2) for this ROI
    a      = entmax_alpha(e)                            a SINGLE pooling, not a mixture

Three parameters (`w_s`, `w_u`, `b`). Initialised at zero so alpha starts at exactly 1.5, i.e.
identical to ASGAP, and any movement away from 1.5 is something the data pushed. GAIN scales the
pre-sigmoid term so the gradient is not vanishingly small on 30 training patients — the fix a327
introduced after the bare scalar in a215 turned out to be inert.

Because alpha now moves freely across the whole (1, 2) interval, this is the sharper test of the
hypothesis: if the physical scale really determines the right sparsity, `w_s` should end up
clearly non-zero and consistent in sign across folds and backbones.

CONTROL: `condition=False` drops both inputs, leaving `alpha = 1 + sigmoid(b)` — a single global
learnable order shared by every ROI. That is the same object a215 already showed to be inert, so
it doubles as a sanity check: if the control's alpha also stays at 1.5, the machinery is behaving,
and any difference between the two arms is attributable to the scale input alone.

`alpha_report()` returns the learned alpha at a range of magnifications, so the mapping can be
read off directly rather than inferred from accuracy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

GAIN = 3.0     # amplifies the pre-sigmoid term; without it the scalar receives a tiny gradient


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, condition: bool = True):
        super().__init__()
        self.condition = bool(condition)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.w_s = nn.Parameter(torch.zeros(()))    # scale coefficient
        self.w_u = nn.Parameter(torch.zeros(()))    # unmeasurable offset
        self.b = nn.Parameter(torch.zeros(()))      # sigmoid(0) = 0.5 -> alpha = 1.5 at init
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()

    def _alpha(self, s: float, u: float) -> torch.Tensor:
        if not self.condition:
            return 1.0 + torch.sigmoid(GAIN * self.b)
        return 1.0 + torch.sigmoid(GAIN * (self.w_s * s + self.w_u * u + self.b))

    @torch.no_grad()
    def alpha_report(self, um_values=(0.27, 0.36, 0.41, 0.54, 0.82, 1.06, 1.35, 2.70, None)) -> dict:
        out = {}
        for v in um_values:
            s, u = ((0.0, 1.0) if v is None else ((np.log10(v) - LOG_REF) / LOG_SCALE, 0.0))
            out["unknown" if v is None else f"{v:.2f}"] = round(float(self._alpha(s, u)), 4)
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


KWARGS = dict(input_dim=1280, num_classes=1, condition=True)
