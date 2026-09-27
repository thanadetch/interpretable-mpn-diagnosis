"""scale_pick_pinned — each MEASURED magnification band commits to one of four orders;
ROIs whose scale bar cannot be read stay at 1.5.

    alpha in {1.00, 1.25, 1.50, 2.00}

    band 0   um/px  < 0.45         ~25-40x    theta[0] -> picks one of the four
    band 1   0.45 <= < 0.70        ~20x       theta[1]
    band 2   um/px >= 0.70         ~4-12x     theta[2]
    scale bar unreadable (47%)                alpha = 1.5, FIXED, no gradient path

WHY THIS IS NOT a415 AGAIN
--------------------------
a415 offered five orders and kept a FOURTH row for the unmeasurable ROIs, so that group — 47% of
the data — competed for the same gradient. It then picked 1.50 in 53 of 60 band x fold cells and
its softmax preferences never left uniform (.1944/.1975/.2076/.2023/.1980 against .2000), i.e. it
expressed no preference at all.

Pinning the unmeasurable group is exactly the change that separated a417 from its control: with
the 47% removed from the learning problem, a417's scale coefficient carried the right sign in 17
of 18 runs against 7 of 18 for a shuffled-scale control, and its alpha span was 5x the control's.
So the discrete-pick design deserves one re-test under the same pinning. Three rows of four
logits, 12 parameters, all zero at initialisation, so nothing is pre-loaded toward any order.

1.75 is dropped at the user's request; the four remaining orders still span softmax (1.0) to
sparsemax (2.0) and keep 1.50, the value ASGAP uses.

A straight-through Gumbel-softmax draws a one-hot selection during training, so the value that
reaches the pooling is a single order while the gradient still reaches all four logits. At
evaluation the sampling is off and the row's argmax is taken, so each band ends up committed to
exactly one order and the model is deterministic.

`alpha_report()` returns each band's chosen order with the softmax preference behind it — the
direct answer to "what alpha does this magnification want", readable without touching accuracy.

CONTROL
-------
`shuffle=True` assigns each measurable ROI a band drawn from the observed band distribution
instead of its own, preserving the marginal and destroying only the correspondence. The
unmeasurable ROIs stay pinned in both arms, so the pair differs in exactly one thing.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

ALPHAS = (1.00, 1.25, 1.50, 2.00)
PINNED_ALPHA = 1.5
TAU = 1.0
_EDGES = [(np.log10(v) - LOG_REF) / LOG_SCALE for v in (0.45, 0.70)]
BAND_NAMES = ("<0.45 (~25-40x)", "0.45-0.70 (~20x)", ">=0.70 (~4-12x)")
UNKNOWN_BAND = 3        # only exists when pin=False


class Model(nn.Module):
    """`pin=True` (a425/a426): three rows, unmeasurable ROIs held at 1.5 with no gradient path.
    `pin=False` (a427/a428): a fourth row, so the unmeasurable 47% learn an order of their own —
    the single-change ablation that measures what the pinning is worth."""

    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0, tau: float = TAU,
                 pin: bool = True):
        super().__init__()
        self.shuffle, self.seed, self.tau = bool(shuffle), int(seed), float(tau)
        self.pin = bool(pin)
        n_bands = 3 if self.pin else 4
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.theta = nn.Parameter(torch.zeros(n_bands, len(ALPHAS)))   # uniform at init
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)
        # observed band distribution, for the shuffled arm. When pin=False the unknown group is
        # one of the bands, so it belongs in the pool too and the marginal stays intact.
        bands = [(UNKNOWN_BAND if v[1] >= 0.5 else int(np.searchsorted(_EDGES, v[0], side="right")))
                 for v in self.bank.table.values()
                 if (not self.pin) or v[1] < 0.5]
        self._pool = np.array(bands, dtype=int) if bands else np.arange(n_bands)

    def _band(self, s: float, u: float) -> Optional[int]:
        if self.pin and u > 0.5:
            return None                                   # unreadable -> pinned
        if self.shuffle:
            return int(self._rng.choice(self._pool))      # keep marginal, break correspondence
        if u > 0.5:
            return UNKNOWN_BAND
        return int(np.searchsorted(_EDGES, s, side="right"))

    def _select(self, band: int) -> torch.Tensor:
        logits = self.theta[band]
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=True)
        return F.one_hot(logits.argmax(), len(ALPHAS)).to(logits.dtype)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        names = list(BAND_NAMES) + ([] if self.pin else ["unknown"])
        out = {}
        for b, name in enumerate(names):
            p = torch.softmax(self.theta[b], dim=0)
            out[name] = {"alpha": ALPHAS[int(self.theta[b].argmax())],
                         "pref": [round(float(x), 3) for x in p]}
        if self.pin:
            out["unknown"] = {"alpha": PINNED_ALPHA, "pref": None}
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        band = self._band(s, u)

        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)

        if band is None:                                   # pinned, ordinary entmax at 1.5
            a = entmax_bisect(e, PINNED_ALPHA)
            z = torch.mv(h.t(), a)
        elif self.training:
            sel = self._select(band)                       # one-hot [4]
            z = 0.0
            a = 0.0
            for k, al in enumerate(ALPHAS):
                ak = F.softmax(e, dim=0) if al <= 1.0 + 1e-6 else entmax_bisect(e, al)
                z = z + sel[k] * torch.mv(h.t(), ak)
                a = a + sel[k] * ak
        else:
            al = ALPHAS[int(self._select(band).argmax())]
            a = F.softmax(e, dim=0) if al <= 1.0 + 1e-6 else entmax_bisect(e, al)
            z = torch.mv(h.t(), a)

        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
