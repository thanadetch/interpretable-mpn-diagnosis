"""scale_two_stage_alpha — pick an order per magnification, then refine it.

    STAGE 1   each measured band commits to one of {1.00, 1.25, 1.50, 2.00} via a
              straight-through Gumbel pick, exactly as a425 does. The unmeasurable band does
              not take part; its starting order is 1.50.

    STAGE 2   the pick is frozen to its argmax and becomes the STARTING POINT of a per-band
              learnable refinement:

                  alpha_b = clip( base_b + R * tanh(GAIN * delta_b),  1.0, 2.0 )

              `delta` is one free scalar per band, zero-initialised, so stage 2 begins at
              exactly the order stage 1 chose. EVERY band keeps its own delta, including the
              unmeasurable one and including bands that happened to pick the same order — two
              bands that both land on 1.50 can still refine to different values.

WHAT THIS IS TESTING
--------------------
a215's learnable alpha is inert: it starts at 1.5 and converges to 1.491-1.505 across every
backbone and fold. The standing explanation has been that 30 training patients cannot move a
single global scalar. This module tests a different explanation the user proposed — that the
scalar does not move because 1.5 is a poor place to start, and that a per-magnification starting
point found first would let the refinement go somewhere.

That is a real open question. If stage 2 pulls alpha meaningfully away from the order stage 1
picked, and does so consistently, then the inertness was about initialisation. If the deltas stay
at zero the way a215's alpha_raw did, the initialisation was never the obstacle.

The switch is by step count rather than by epoch, so the trainer needs no modification: the
module counts its own training forward passes and flips at STAGE1_STEPS, which is about eight
epochs at this cohort's ~900 bags per epoch.

R bounds the refinement at 0.25, half the gap to the neighbouring grid point, so stage 2 can
adjust an order without silently jumping to a different one. GAIN amplifies the pre-tanh term;
without it a bare scalar receives too small a gradient on 30 patients, the failure a327 diagnosed.

CONTROL: `shuffle=True` draws each ROI's band from the observed band distribution rather than
its own, preserving the marginal and destroying only the correspondence. Both stages run
identically in both arms.
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
N_BANDS = 4                       # 3 measured + unknown; unknown starts at 1.5, no stage-1 pick
UNKNOWN_BAND = 3
UNKNOWN_BASE = 1.5
TAU = 1.0
R = 0.25                          # refinement radius, half the gap to the neighbouring order
GAIN = 3.0
ALPHA_FLOOR = 1.02                # alpha = 1 is the boundary of the entmax domain: a band that
STAGE2_MIN_BASE = 1.05            # picks 1.00 sits on the clamp and receives no gradient at all,
                                  # so stage 2 starts such a band at 1.05 instead. entmax at 1.05
                                  # differs from softmax by <0.04, so the pooling is still flat.
STAGE1_STEPS = 7200               # ~8 epochs at ~900 bags/epoch
_EDGES = [(np.log10(v) - LOG_REF) / LOG_SCALE for v in (0.45, 0.70)]
BAND_NAMES = ("<0.45 (~25-40x)", "0.45-0.70 (~20x)", ">=0.70 (~4-12x)", "unknown")


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0, tau: float = TAU,
                 stage1_steps: int = STAGE1_STEPS):
        super().__init__()
        self.shuffle, self.seed, self.tau = bool(shuffle), int(seed), float(tau)
        self.stage1_steps = int(stage1_steps)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.theta = nn.Parameter(torch.zeros(3, len(ALPHAS)))   # stage 1, measured bands only
        self.delta = nn.Parameter(torch.zeros(N_BANDS))          # stage 2, one per band
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.bank = _ScaleFeatureBank()
        self._rng = np.random.default_rng(self.seed)
        bands = [(UNKNOWN_BAND if v[1] >= 0.5 else int(np.searchsorted(_EDGES, v[0], side="right")))
                 for v in self.bank.table.values()]
        self._pool = np.array(bands, dtype=int) if bands else np.arange(N_BANDS)

    # ---- stages -------------------------------------------------------------------------
    @property
    def in_stage2(self) -> bool:
        return int(self.step) >= self.stage1_steps

    def _band(self, s: float, u: float) -> int:
        if self.shuffle:
            return int(self._rng.choice(self._pool))
        if u > 0.5:
            return UNKNOWN_BAND
        return int(np.searchsorted(_EDGES, s, side="right"))

    def _base(self, band: int):
        """Stage-1 order for this band: a sampled one-hot early on, its argmax afterwards."""
        if band == UNKNOWN_BAND:
            return torch.tensor(UNKNOWN_BASE, device=self.delta.device, dtype=self.delta.dtype), None
        logits = self.theta[band]
        if self.training and not self.in_stage2:
            sel = F.gumbel_softmax(logits, tau=self.tau, hard=True)
            return None, sel                                   # value carried by the mixture
        return torch.tensor(ALPHAS[int(logits.argmax())],
                            device=self.delta.device, dtype=self.delta.dtype), None

    def _refine(self, base, band: int):
        base = torch.clamp(base, min=STAGE2_MIN_BASE)   # keep the start off the alpha=1 boundary
        a = base + R * torch.tanh(GAIN * self.delta[band])
        return torch.clamp(a, ALPHA_FLOOR, 2.0)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        out = {}
        for b, name in enumerate(BAND_NAMES):
            if b == UNKNOWN_BAND:
                base = UNKNOWN_BASE
                pref = None
            else:
                base = ALPHAS[int(self.theta[b].argmax())]
                pref = [round(float(x), 3) for x in torch.softmax(self.theta[b], dim=0)]
            fin = float(np.clip(max(base, STAGE2_MIN_BASE) + R * np.tanh(GAIN * float(self.delta[b])),
                                ALPHA_FLOOR, 2.0))
            out[name] = {"stage1": base, "stage2": round(fin, 4),
                         "shift": round(fin - base, 4), "delta": round(float(self.delta[b]), 5),
                         "pref": pref}
        out["_step"] = int(self.step)
        out["_reached_stage2"] = bool(self.in_stage2)
        return out

    # ---- forward ------------------------------------------------------------------------
    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        if self.training:
            self.step += 1
        s, u = self.bank.feat(features)
        band = self._band(s, u)

        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)

        base, sel = self._base(band)
        if sel is not None:
            # stage 1 on a measured band: mixture so the gradient reaches every logit, while the
            # hard one-hot means a single order actually determines the value
            z = 0.0
            a = 0.0
            for k, al in enumerate(ALPHAS):
                al_r = float(np.clip(al, 1.0, 2.0))
                ak = F.softmax(e, dim=0) if al_r <= 1.0 + 1e-6 else entmax_bisect(e, al_r)
                z = z + sel[k] * torch.mv(h.t(), ak)
                a = a + sel[k] * ak
        else:
            alpha = self._refine(base, band)
            a = entmax_bisect(e, alpha)          # floored above 1, so this always has a gradient
            z = torch.mv(h.t(), a)

        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
