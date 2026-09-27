"""a60 — Differentiable soft-RANK severity pooling (L-estimator of the
per-patch fibrosis-severity distribution).

Hypothesis (H_softrank_severity)
--------------------------------
Reticulin grade is a *diffuse, ordinal* property: it is the typical
severity of the fibre meshwork across the marrow, not the single worst
patch (top-k / argmax, refuted) nor the overall mean (a56/a57, mean-pool).
A robust, grading-aligned summary of a distribution is an **L-estimator**:
a weighted average of the *order statistics* where the weights depend on
each sample's *rank within the bag*, not on its raw value. Emphasising the
**upper-middle** severities (e.g. the ~60-80th rank-percentile) reads the
"how fibrotic is the genuinely-affected tissue, ignoring a few clean
outliers and not letting one hot patch dominate" — which is what an ordinal
G0..G3 call integrates. This is exactly a trimmed/winsorised mean made
SMOOTH and LEARNABLE (the trim location/width are learned), so it is a
strict generalisation of mean-pool that can also down-weight the bottom and
the extreme top.

Mechanism (exact)
-----------------
1. h_i = Dropout(ReLU(Linear(1280->128) f_i))            # baseline encoder
2. s_i = Linear(32->1)(ReLU(Linear(128->32) h_i))        # per-patch severity
        (NONLINEAR, pre-aggregation — like a56; legal under the contract)
3. Differentiable normalized soft-rank of each patch WITHIN the bag:
        r_i = (1/(N-1)) * Σ_{j≠i} sigmoid( (s_i - s_j) / β )   ∈ [0,1]
   r_i is the soft fraction of patches that s_i exceeds → a within-bag
   rank-percentile. β (softplus-parameterised) controls comparison
   sharpness. r_i is BAG-SIZE-INVARIANT (normalized by N-1) and
   PERMUTATION-INVARIANT (symmetric pairwise sum).
4. Smooth rank-emphasis weight (a learnable Gaussian bump ON THE RANK axis,
   NOT on the value and NOT on ||h||):
        a_i = exp( -(r_i - μ)^2 / (2 · σ_w^2) )
        w_i = a_i / Σ_j a_j                                (normalize)
   μ = sigmoid(μ_raw) ∈ (0,1) is the learned emphasis quantile (init ≈ 0.7,
   upper-mid); σ_w = softplus(σ_raw) is the learned band width. As σ_w → ∞
   this reduces to the plain mean (a57/mean-pool) — so a60 ⊇ mean-pool and
   the bump can only help if rank-position carries signal.
5. Bag severity (rank-weighted mean of the per-patch severities):
        ŝ = Σ_i w_i · s_i
        ŷ = clamp(ŝ, 0, 3)                                 # linear/unbounded readout
   No σ/softmax GATE multiplies the bag representation by anything — w is a
   convex average over the SAME severities, an L-estimator, not a gate on a
   bounded head.

Why this is GENUINELY DIFFERENT (not a refuted dead-end)
--------------------------------------------------------
- NOT the a43..a51 / a45 rank-norm family: those rank patches by **||h||**
  (L2 norm), which the diagnostics proved grade-uninformative. a60 ranks by
  a **learned fibrosis severity** s_i; ||h|| is never computed or used.
- NOT a49/a50 score-SOFTMAX (DE32): those put softmax(s/τ) over the raw
  score (an attention-concentration that down-weights low scores
  monotonically). a60 weights by a NON-MONOTONE bump on the *rank*; it can
  emphasise the middle/upper-middle and SUPPRESS the top — the opposite of
  softmax/argmax. No softmax-over-a-direction.
- NOT coverage/extent/threshold/fraction (a52/a54, refuted 3×): there is no
  sigmoid-of-(score−τ) counted into a fraction; the readout is a
  rank-weighted MEAN SEVERITY, not "fraction of fibre-positive patches".
- NOT top-k / argmax / hard trim (DE06, DE16): the rank weighting is a
  SMOOTH differentiable Gaussian-on-rank; no patches are hard-selected or
  hard-dropped, and the trim location/width are LEARNED, not fixed.
- NOT mean-pool (a56/a57): unequal, rank-dependent weights on a NONLINEAR
  per-patch severity → depends on the *shape* of the severity distribution,
  not just its mean. (a57 is the σ_w→∞, linear-severity collapse of a60.)
- NOT a σ-gate head (DE11-13): the only nonlinearities are per-patch
  (ReLU MLP for s_i, and the rank-bump weighting), all PRE-aggregation; the
  bag readout is the (unbounded) weighted mean of severities, clamped only
  at the very end. No sigmoid/softmax sits between the bag rep and ŷ.

Ablation companion: a61_softrank_meanpool — set σ_raw very large (bump → flat)
so w_i ≡ 1/N and ŷ = mean_i s_i (= a56). a60 vs a61 isolates the active
ingredient = "does a LEARNED rank-emphasis (L-estimator) beat the plain mean
of the same per-patch severities?".

Kill criterion: abandon if a60 val_qwk < 0.78 at seed=2 AND a60 ≤ a61
(rank-emphasis adds nothing over mean). DoD = multi-seed audit {0,1,2,3,42}.

Param count (input_dim=1280, hidden=128, sev_hidden=32):
    bottleneck Linear(1280,128)+b = 163,968
    severity   Linear(128,32)+b   =   4,128
               Linear(32,1)+b     =      33
    beta_raw                      =       1   (soft-rank temperature)
    mu_raw                        =       1   (emphasis quantile)
    sigma_raw                     =       1   (band width)
    -----------------------------------------
    total                         = 168,132   (< ~200K; +3 over a56)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        sev_hidden: int = 32,
        beta_init: float = 0.5,        # softplus(beta_raw) → soft-rank temperature
        mu_init: float = 0.7,          # emphasis quantile in (0,1); 0.7 = upper-mid
        sigma_init: float = 0.25,      # rank-bump width (softplus-parameterised)
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-patch NONLINEAR severity head (pre-aggregation; legal).
        self.severity = nn.Sequential(
            nn.Linear(hidden_dim, sev_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(sev_hidden, 1),
        )
        # Start mid-grade so the weighted-mean severity sits inside [0,3].
        with torch.no_grad():
            self.severity[-1].bias.fill_(1.5)

        # Soft-rank temperature β = softplus(beta_raw)  (>0).
        self.beta_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(beta_init)))
        )
        # Emphasis quantile μ = sigmoid(mu_raw) ∈ (0,1).
        self.mu_raw = nn.Parameter(
            torch.tensor(_inv_sigmoid(float(mu_init)))
        )
        # Band width σ_w = softplus(sigma_raw)  (>0).
        self.sigma_raw = nn.Parameter(
            torch.tensor(_inv_softplus(float(sigma_init)))
        )

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        h = self.bottleneck(features)              # [N, hidden]
        s = self.severity(h).squeeze(-1)           # [N] per-patch severity
        N = s.shape[0]

        if N == 1:
            # Degenerate single-patch bag: rank/weight are trivial.
            y = s.view(1)
            if self.clamp_output:
                y = y.clamp(0.0, 3.0)
            if return_attention:
                return y, torch.ones(1, device=s.device), None
            return y, None, None

        # --- Differentiable normalized soft-rank within the bag ---
        beta = F.softplus(self.beta_raw).clamp(min=1e-3)
        # Pairwise comparisons: P[i,j] = sigmoid((s_i - s_j)/beta)
        diff = (s.unsqueeze(1) - s.unsqueeze(0)) / beta        # [N, N]
        P = torch.sigmoid(diff)                                # [N, N]
        # Exclude self-comparison (sigmoid(0)=0.5 each); normalize by N-1.
        rank = (P.sum(dim=1) - 0.5) / (N - 1)                  # [N] in [0,1]

        # --- Smooth Gaussian rank-emphasis weight (learned location/width) ---
        mu = torch.sigmoid(self.mu_raw)                        # () in (0,1)
        sigma = F.softplus(self.sigma_raw).clamp(min=1e-3)     # () > 0
        log_a = -((rank - mu) ** 2) / (2.0 * sigma * sigma)    # [N]
        w = torch.softmax(log_a, dim=0)                        # [N] normalized

        # --- Rank-weighted mean of the SAME per-patch severities (L-estimator) ---
        s_hat = (w * s).sum().view(1)                          # [1]
        y = s_hat
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, w, None
        return y, None, None


def _inv_softplus(y: float) -> float:
    """x such that softplus(x) = y, for y > 0."""
    return math.log(math.expm1(y)) if y < 20.0 else y


def _inv_sigmoid(p: float) -> float:
    """x such that sigmoid(x) = p, for p in (0,1)."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    sev_hidden=32,
    beta_init=0.5,
    mu_init=0.7,
    sigma_init=0.25,
    clamp_output=True,
)
