"""a56 — Per-patch severity pooling (pool SCORES, not features).

Angle: instead of pooling patch FEATURES (baseline: attention-weighted mean
of h, then one head) or a single linear projection (a53 ~ mean-pool), give
EACH patch its own nonlinear fibrosis-severity estimate and AVERAGE those
estimates. This is the literal "overall density" reading: score every patch
for how fibrotic it looks, then take the mean severity over the marrow.

    h_i = Dropout(ReLU(Linear(1280->128) f_i))         # baseline encoder
    g_i = Linear(32->1)(ReLU(Linear(128->32) h_i))     # per-patch severity (NONLINEAR)
    y   = clamp(mean_i g_i, 0, 3)

Key point — why this is NOT mean-pool (a53/DE) : because the per-patch map is
NONLINEAR, mean_i MLP(h_i) != MLP(mean_i h_i). The bag readout depends on the
DISTRIBUTION of per-patch severities (a bag with a few very-severe patches and
many clean ones differs from a uniformly-mild bag with the same mean feature).
mean-pool / a53 collapse that distribution; this keeps it.

Pathology rationale: matches "assign each region a fibrosis severity, average
over the ROI" — diffuse, bag-wide, permutation- and size-invariant. No
attention concentration, no norm, no coverage/threshold (the refuted family:
C1 a52/a53, C2 a54/a55 all had coverage <= mean). The novelty here is the
NONLINEAR per-patch severity, an orthogonal lever to "coverage vs mean".

Why not a dead-end:
  - NOT mean-pool / a53 (those are LINEAR per-patch -> reduce to MLP(mean));
    here the per-patch head is a 2-layer MLP (nonlinear).
  - NOT coverage/extent (a52/a54, refuted): no sigmoid threshold / fraction;
    g_i is an unbounded severity, averaged directly.
  - NOT attention/top-k/softmax: aggregation is an unweighted mean, no patch
    weighting at all.
  - NOT norm-based (a45): severity is a learned MLP of h, ||h|| never used.
  - DE01 (DeepSet) adjacency, stated honestly: this IS a low-capacity
    phi-mean-rho set function. DE01 found DeepSet overfit, but on the LEGACY
    uni2/mean_pool baseline with much higher capacity; NOVELTY_NOTES notes
    DE01 "may not hold on the larger full-data split". Here phi is a tiny
    128->32->1 head (~4K params over the baseline encoder) and rho is the
    parameter-free mean -> a deliberately low-capacity DeepSet-lite.
  - No sigmoid/softmax GATE between bag rep and scalar (DE11-13): the only
    nonlinearity is the per-patch ReLU MLP, PRE-aggregation; readout = mean
    then clamp, linear/unbounded.

Ablation companion: a57_perpatch_linear_pool.py — flips severity_mode to
'linear' (g_i = Linear(128->1) h_i), so y = clamp(mean_i <h_i,w>+b) =
clamp(<mean_i h_i, w>+b) = mean-pool + linear head. Removes EXACTLY the
per-patch nonlinearity (the active ingredient). a56 vs a57 tests: "does a
NONLINEAR per-patch severity (distribution-aware) beat a LINEAR one
(mean-pool)?"

Kill criterion: abandon if a56 val_qwk < 0.78 at seed=2 AND a56 <= a57
(nonlinear per-patch adds nothing over mean-pool). DoD = multi-seed audit.

Param count (input_dim=1280, hidden=128, sev_hidden=32):
    bottleneck Linear(1280,128)+b = 163,968
    severity   Linear(128,32)+b    =   4,128
               Linear(32,1)+b      =      33
    -------------------------------------------
    total (mlp)                    = 168,129   (< 197,250 baseline)
    a57 ablation (linear severity) = 164,097
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        severity_mode: str = "mlp",     # "mlp" (a56 main) | "linear" (a57 ablation)
        sev_hidden: int = 32,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert severity_mode in ("mlp", "linear"), severity_mode
        self.severity_mode = severity_mode
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-patch severity head.
        if severity_mode == "mlp":
            self.severity = nn.Sequential(
                nn.Linear(hidden_dim, sev_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(sev_hidden, 1),
            )
            last = self.severity[-1]
        else:  # linear -> reduces to mean-pool + linear head
            self.severity = nn.Linear(hidden_dim, 1)
            last = self.severity
        # Start mid-grade so mean severity sits in the interior of [0,3].
        with torch.no_grad():
            last.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        h = self.bottleneck(features)             # [N, hidden]
        g = self.severity(h).squeeze(-1)          # [N] per-patch severity
        y = g.mean().view(1)                      # scalar bag severity
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)
        if return_attention:
            return y, g, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    severity_mode="mlp",     # a56 main = nonlinear per-patch severity
    sev_hidden=32,
    clamp_output=True,
)
