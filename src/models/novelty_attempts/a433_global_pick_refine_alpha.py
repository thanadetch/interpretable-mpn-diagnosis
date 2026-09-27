"""a433 — Global discrete alpha PICK, then continuous refinement (user's design).

Two stages, no external covariate anywhere -- the model chooses its own sparsity level.

    STAGE 1  a straight-through Gumbel-softmax picks ONE order from
             ALPHAS = {1.00, 1.25, 1.50, 1.75, 2.00}.  All five logits start at 0, so nothing is
             pre-loaded toward any order, and the gradient reaches all five while the value that
             actually reaches the pooling is a single alpha.

    STAGE 2  the pick is frozen to its argmax and becomes the STARTING POINT of a continuous
             learnable refinement:

                 alpha = clip( base + R * tanh(GAIN * delta), 1.02, 2.0 )

             `delta` is one free scalar, zero-initialised, so stage 2 begins exactly at the order
             stage 1 chose -- the "checkpoint" the user described.

WHY THIS IS NOT ANY EARLIER MODULE
----------------------------------
a415/a425/a427/a429 also pick from a discrete set, but every one of them conditions the pick on the
ROI's magnification band, so the pick is tied to a covariate that a 30-run sweep showed is not
useful. a289 mixes four FIXED orders (1.1-1.55) with learned weights rather than committing to one,
and its grid excludes the sparse region. a239/a240 are continuous-alpha runs whose init the
experimenter chose by hand. Nothing so far lets the model choose its own order from the full grid.

WHAT IT TESTS
-------------
A fixed-alpha sweep at seed 2 puts the best order at ~1.75-1.80 on Virchow2 and TITAN and at
~1.00-1.25 on UNI2-h, and an init sweep (a239/a240) showed a continuous alpha does not travel: it
stays within 0.02 of wherever it starts. If the discrete stage is genuinely able to learn, its
argmax should land near 1.75 on Virchow2/TITAN and near 1.00-1.25 on UNI2-h, WITHOUT being told.
If instead the softmax preferences stay flat and the pick is arbitrary, then alpha cannot be learned
in any parameterisation and must be set by sweep -- which is itself the design justification for a
fixed order.

R = 0.125 is half the grid spacing, so stage 2 can adjust an order without silently jumping to a
neighbouring one. STAGE2_MIN_BASE keeps a pick of 1.00 off the entmax boundary, where the clamp
would otherwise zero the refinement gradient (the failure a429 diagnosed).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect

ALPHAS = (1.00, 1.25, 1.50, 1.75, 2.00)
TAU = 1.0
R = 0.125
GAIN = 3.0
ALPHA_FLOOR = 1.02
STAGE2_MIN_BASE = 1.05
STAGE1_STEPS = 7200          # ~8 epochs at ~900 bags/epoch


class Model(nn.Module):
    """`refine=False` (a434) stops after stage 1 — the ablation that isolates the refinement."""

    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, tau: float = TAU, refine: bool = True,
                 stage1_steps: int = STAGE1_STEPS):
        super().__init__()
        self.tau, self.refine = float(tau), bool(refine)
        self.stage1_steps = int(stage1_steps)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.theta = nn.Parameter(torch.zeros(len(ALPHAS)))   # stage 1: uniform at init
        self.delta = nn.Parameter(torch.zeros(()))            # stage 2: refinement
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @property
    def in_stage2(self) -> bool:
        return self.refine and int(self.step) >= self.stage1_steps

    def _refined(self, base: float) -> torch.Tensor:
        b = torch.clamp(torch.as_tensor(base, dtype=self.delta.dtype, device=self.delta.device),
                        min=STAGE2_MIN_BASE)
        return torch.clamp(b + R * torch.tanh(GAIN * self.delta), ALPHA_FLOOR, 2.0)

    @torch.no_grad()
    def alpha_report(self) -> dict:
        p = torch.softmax(self.theta, dim=0)
        k = int(self.theta.argmax())
        base = ALPHAS[k]
        out = {"picked_alpha": base,
               "pref": {f"{a:.2f}": round(float(v), 4) for a, v in zip(ALPHAS, p)},
               "pref_margin": round(float(p.max() - p.min()), 4),
               "delta": round(float(self.delta), 5),
               "_step": int(self.step), "_stage2": bool(self.in_stage2)}
        out["final_alpha"] = round(float(self._refined(base)), 4) if self.refine else base
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        if self.training:
            self.step += 1
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)

        if self.training and not self.in_stage2:
            sel = F.gumbel_softmax(self.theta, tau=self.tau, hard=True)
            z = 0.0
            a = 0.0
            for k, al in enumerate(ALPHAS):
                ak = F.softmax(e, dim=0) if al <= 1.0 + 1e-6 else entmax_bisect(e, al)
                z = z + sel[k] * torch.mv(h.t(), ak)
                a = a + sel[k] * ak
        else:
            base = ALPHAS[int(self.theta.argmax())]
            if self.refine:
                a = entmax_bisect(e, self._refined(base))
            else:
                a = F.softmax(e, dim=0) if base <= 1.0 + 1e-6 else entmax_bisect(e, base)
            z = torch.mv(h.t(), a)

        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, refine=True)
