"""scale_formula_v2 — three FORMS for alpha as a function of measured um/px, unknowns pinned at 1.5.

a417 showed the model genuinely reads the scale (w_s carries the right sign in 17 of 18 runs
versus 7 of 18 for its shuffled control) but only moves alpha across a span of 0.02-0.17 out of
the 1.0 that is available. Near alpha = 1.5 the sigmoid form is close to linear, so

    alpha ~= 1.5 + (GAIN / 4) * w_s * s ,   s spanning about 2.86 across 0.27-2.70 um/px
    span(alpha) ~= 2.15 * |w_s|

which says the limit is the size of the learned coefficient, not the shape of the curve. These
three forms separate the two explanations.

    "gain"    alpha = 1 + sigmoid(GAIN * (w_s*s + b))          same as a417 with a larger GAIN
    "linear"  alpha = clamp(1.5 + w_s*s + b, 1.0, 2.0)         no saturation at all, slope free
    "quad"    alpha = 1 + sigmoid(GAIN * (w1*s + w2*s^2 + b))  may peak in the middle of the range

THE DECISIVE ONE IS "gain"
--------------------------
Raising GAIN multiplies the alpha movement produced by a given w_s. If the span grows with GAIN,
the earlier runs were held back by the parameterisation and the axis is still open. If instead
w_s shrinks by the same factor and the span is unchanged, the optimiser was already sitting where
it wants to be and the narrow span is a property of the loss, not of the formula — which closes
the axis with a much stronger argument than "we tried and it did not help".

"linear" removes the saturating envelope entirely, so nothing bounds the slope except the clamp
at the ends. "quad" allows a non-monotone shape, which a413 hinted at: its two zoomed-in bands
landed within 0.005 of each other while the zoomed-out band sat 0.06 lower, a step rather than a
ramp.

Every form starts at exactly alpha = 1.5 (all coefficients zero) and every form pins ROIs whose
scale bar cannot be read at 1.5 with no gradient path, so the 47% of unmeasurable ROIs cannot
contribute to what is learned about scale. That pinning is what made a417 separate from its
control in the first place.

CONTROL: `shuffle=True` permutes which measured scale each ROI receives, preserving the marginal
distribution of scales and destroying only the ROI-to-scale correspondence. Unknowns stay pinned
in both arms.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

PINNED_ALPHA = 1.5
UM_GRID = (0.27, 0.36, 0.54, 1.06, 2.70)


def _s_of(um: float) -> float:
    return (np.log10(um) - LOG_REF) / LOG_SCALE


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0,
                 form: str = "gain", gain: float = 10.0):
        super().__init__()
        assert form in ("gain", "linear", "quad")
        self.form, self.gain = form, float(gain)
        self.shuffle, self.seed = bool(shuffle), int(seed)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.w_s = nn.Parameter(torch.zeros(()))
        self.w_q = nn.Parameter(torch.zeros(()))      # used only by "quad"
        self.b = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)
        self._pool = np.array([v[0] for v in self.bank.table.values() if v[1] < 0.5], dtype=float)

    def _alpha_from_s(self, s):
        if self.form == "linear":
            return torch.clamp(1.5 + self.w_s * s + self.b, 1.0, 2.0)
        z = self.w_s * s + self.b
        if self.form == "quad":
            z = z + self.w_q * (s * s)
        return 1.0 + torch.sigmoid(self.gain * z)

    def _alpha(self, s: float, u: float):
        if u > 0.5:                       # scale bar unreadable -> pinned, no gradient path
            return torch.tensor(PINNED_ALPHA, device=self.w_s.device, dtype=self.w_s.dtype)
        if self.shuffle and len(self._pool):
            s = float(self._rng.choice(self._pool))
        return self._alpha_from_s(s)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        out = {f"{v:.2f}": round(float(self._alpha_from_s(_s_of(v))), 4) for v in UM_GRID}
        out.update(unknown=PINNED_ALPHA, form=self.form, gain=self.gain,
                   w_s=round(float(self.w_s), 5), w_q=round(float(self.w_q), 5),
                   b=round(float(self.b), 5))
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


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False, form="gain")
