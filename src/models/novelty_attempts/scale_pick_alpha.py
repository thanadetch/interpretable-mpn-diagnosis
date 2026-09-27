"""scale_pick_alpha — each magnification band LEARNS WHICH of five allowed orders to use.

WHAT IS BEING LEARNED
---------------------
The allowed orders are exactly the five the user specified:

    alpha in {1.00, 1.25, 1.50, 1.75, 2.00}

and each scale band keeps a row of five logits over them:

    band 0   um/px  < 0.45          ~25-40x   theta[0] -> picks one of the five
    band 1   0.45 <= < 0.70         ~20x      theta[1]
    band 2   um/px >= 0.70          ~4-12x    theta[2]
    band 3   scale bar unreadable             theta[3]

The forward pass uses ONE order, not a blend: a straight-through Gumbel-softmax draws a one-hot
selection, so the value that reaches the pooling is a single alpha while the gradient still
reaches all five logits. At evaluation the sampling is switched off and the row's argmax is taken,
so the model is deterministic and each band ends up committed to exactly one of the five values.

That is the difference from the two earlier attempts. `scale_gated_alpha` (a409) mixed all five
and its effective order never left 1.478-1.518 because an average cannot do otherwise;
`scale_direct_alpha` (a411) freed the range but forced one monotone function of log scale. Here
the choice is discrete, per band, and unconstrained in order.

All logits start at zero, i.e. a uniform preference over the five, so nothing is pre-loaded toward
1.5. TAU controls how sharp the relaxation is during training.

`alpha_report()` returns each band's chosen order together with the softmax preference behind it,
which is the direct answer to "what alpha does each magnification want" and is readable without
looking at accuracy at all.

CONTROL
-------
`shuffle=True` keeps the four rows and the band sizes but assigns bands to ROIs at random, so the
model can still commit four different orders and simply cannot align them with the true
magnification. a407/a408 already showed that merely having several orders available lifts G1, so
this row is what separates "the scale matters" from "having a choice matters".
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .scale_gated_alpha import LOG_REF, LOG_SCALE, _ScaleFeatureBank

ALPHAS = (1.00, 1.25, 1.50, 1.75, 2.00)
N_BANDS = 4
TAU = 1.0
_EDGES = [(np.log10(v) - LOG_REF) / LOG_SCALE for v in (0.45, 0.70)]
BAND_NAMES = ("<0.45 (~25-40x)", "0.45-0.70 (~20x)", ">=0.70 (~4-12x)", "unknown")


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0, tau: float = TAU):
        super().__init__()
        self.shuffle, self.seed, self.tau = bool(shuffle), int(seed), float(tau)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.theta = nn.Parameter(torch.zeros(N_BANDS, len(ALPHAS)))   # uniform preference at init
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)

    def _band(self, s: float, u: float) -> int:
        if self.shuffle:
            return int(self._rng.integers(0, N_BANDS))
        if u > 0.5:
            return 3
        return int(np.searchsorted(_EDGES, s, side="right"))

    def _select(self, band: int) -> torch.Tensor:
        """One-hot over the five orders: sampled + straight-through in training, argmax at eval."""
        logits = self.theta[band]
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=True)
        return F.one_hot(logits.argmax(), len(ALPHAS)).to(logits.dtype)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        out = {}
        for b, name in enumerate(BAND_NAMES):
            p = torch.softmax(self.theta[b], dim=0)
            out[name] = {"alpha": ALPHAS[int(self.theta[b].argmax())],
                         "pref": [round(float(x), 3) for x in p]}
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        sel = self._select(self._band(s, u))                 # one-hot [5]

        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)

        if self.training:
            # every order is needed so the gradient can reach all five logits; the one-hot makes
            # the VALUE equal to the selected order alone
            z = 0.0
            amix = 0.0
            for k, al in enumerate(ALPHAS):
                a = F.softmax(e, dim=0) if al <= 1.0 + 1e-6 else entmax_bisect(e, al)
                z = z + sel[k] * torch.mv(h.t(), a)
                amix = amix + sel[k] * a
        else:
            k = int(sel.argmax())
            al = ALPHAS[k]
            amix = F.softmax(e, dim=0) if al <= 1.0 + 1e-6 else entmax_bisect(e, al)
            z = torch.mv(h.t(), amix)

        y = self.classifier(z).view(-1)
        if return_attention:
            return y, amix, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
