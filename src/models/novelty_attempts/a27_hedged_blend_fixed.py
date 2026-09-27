"""a27 - hedged prediction-level blend: 0.5*a17 + 0.5*mean_pool (H14, batch 14 main).

Hypothesis (H14_prediction_level_hedged_blend): the a17 + locked-baseline
diagnostics (`results/diagnostics/.../a17_*/summary.md`) showed that
a17 loses to the locked baseline on 58 val ROIs while winning on 78 -
a Pareto trade, not a domination. A late-stage, prediction-level blend
between a17's mechanism-stack and a pure mean-pool branch should
recover the bags a17's coverage prior steers wrong, without giving up
the bags it gets right.

This is mechanistically distinct from `DE17_attn_mean_concat_ensemble`
(batch 7, a13/a14). DE17 concatenated attention-pool and mean-pool in
*feature space* and fed the 256-d vector through a Linear head; the
head can route around the mean branch by zeroing those weights, which
is exactly what a13/a14 showed (within 0.0014 val of each other). A
hard *prediction-level* blend cannot be routed around: with `alpha`
fixed at 0.5, the mean-pool head's scalar must contribute 50% to
every prediction. That is a structural regulariser, not a soft hint
the optimiser can ignore.

Also addresses the diagnostics "norm-rank family saturated" finding
(p90(||h||) varies by only 1.3% across grades on Virchow2). The mean
branch deliberately uses NO norm-based reweighting - it is a pure
direction-only readout - so it cannot be hurt by the same flat-norm
signal that retroactively explains the DE15/DE19 plateau.

Mechanism (exact):
    y_a17,  attn  = A17Model(features, clamp_output=False)        # raw scalar in R
    y_mean        = Linear(1280, 1)(mean_i features_i)            # raw scalar in R
    y             = alpha*y_a17 + (1 - alpha)*y_mean               # fixed alpha=0.5 here
    y             = clamp(y, 0, 3)                                 # only the blend is clamped

Ablation companion: `a28_hedged_blend_learned.py` - same architecture
but `alpha = sigmoid(gamma)` learnable scalar (initialised at 0.5).
Tells us whether the *blending* is the active ingredient or whether
learned alpha drifts to 1 (use a17 only) or 0 (use mean only).

Kill criterion: abandon H14 if BOTH a27 and a28 val_qwk < a17's
reproducible 0.7970 at seed=2.

Param count (input_dim=1280, hidden_dim=128):
    A17Model (full)                              = 197,254
    mean_head Linear(1280, 1) + bias             =   1,281
    alpha (buffer, frozen at 0.5, not counted)   =       0
    -----------------------------------------------------
    total trainable                              = 198,535
    (vs ABMIL 197,250 - i.e. +1,285 params, ~0.7% over baseline)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .a17_coverage_length_norm import Model as _A17Model


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,
        learned_alpha: bool = False,
        alpha_init: float = 0.5,
        # a17 sub-module hyperparameters (kept at a17 defaults so a27 reduces
        # to the same a17 stack at alpha=1).
        tau_init: float = 1.0,
        beta_init: float = 1.0,
        alpha17_init: float = 1e-3,
        c0_init: float = 6.324555,
        use_coverage: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output
        self.learned_alpha = learned_alpha

        # Branch A: the a17 stack with internal clamping disabled so the
        # blend receives raw (unclamped) scalars and gradients flow cleanly.
        self.a17 = _A17Model(
            input_dim=input_dim,
            num_classes=1,
            hidden_dim=hidden_dim,
            dropout=dropout,
            clamp_output=False,
            tau_init=tau_init,
            beta_init=beta_init,
            alpha_init=alpha17_init,
            c0_init=c0_init,
            use_coverage=use_coverage,
        )

        # Branch B: pure mean-pool readout. No bottleneck, no attention,
        # no norm-based reweighting -> cannot be hurt by the flat-norm
        # signal on Virchow2. Acts as a structural regulariser via the
        # fixed-alpha blend below.
        self.mean_head = nn.Linear(input_dim, 1)

        # Blend coefficient: fixed 0.5 buffer (a27) or learnable scalar (a28).
        if learned_alpha:
            eps = 1e-6
            a = max(min(float(alpha_init), 1.0 - eps), eps)
            gamma_init = math.log(a / (1.0 - a))  # so sigmoid(gamma_init) = a
            self.gamma = nn.Parameter(torch.tensor(gamma_init))
        else:
            self.register_buffer("alpha_const", torch.tensor(float(alpha_init)))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # Branch A: a17 raw scalar (no clamp).
        y_a17, attn, _ = self.a17(features, return_attention=True)
        # Branch B: mean-pool raw scalar.
        bag_mean = features.mean(dim=0)                       # [D]
        y_mean = self.mean_head(bag_mean.unsqueeze(0)).squeeze(0)  # [1]

        # Blend.
        if self.learned_alpha:
            alpha = torch.sigmoid(self.gamma)
        else:
            alpha = self.alpha_const
        y = alpha * y_a17 + (1.0 - alpha) * y_mean             # [1]

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    learned_alpha=False,    # a27 main: fixed 50/50 blend (no new trainable param)
    alpha_init=0.5,
    tau_init=1.0,
    beta_init=1.0,
    alpha17_init=1e-3,
    c0_init=6.324555,
    use_coverage=True,
)

