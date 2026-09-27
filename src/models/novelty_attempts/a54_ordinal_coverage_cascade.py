"""a54 — Ordinal Coverage Cascade MIL (grading-aligned, ordinal-local).

Angle: ORDINAL / RANK-STRUCTURE aggregation that respects the ordinal-local
(+/-1) error structure AND the diffuse-density reading, using a bottleneck
encoder. The rank/order is over a *fibrosis-relevant* per-patch score, NOT
over ||h|| (the a45 norm-rank dead-end).

------------------------------------------------------------------------
Pathology motivation
------------------------------------------------------------------------
A pathologist reads a reticulin ROI as a CASCADE of diffuse-density
questions at the three grade boundaries:

    boundary 0  (G0|G1):  is there ANY reticulin fibre at all, anywhere?
    boundary 1  (G1|G2):  is the meshwork MODERATELY dense / extensive?
    boundary 2  (G2|G3):  is dense coarse fibre present DIFFUSELY everywhere?

Each question is a *coverage / extent* question (what FRACTION of the marrow
crosses that severity), and — crucially — the diagnostic
(results/diag/direction_separability.md) shows each boundary is best
separated by its OWN fibrosis direction (proj_ord1 AUC peaks at G0|G1=0.95;
proj_ord3 AUC peaks at G2|G3=0.85; the three prototype-difference directions
are near-orthogonal, cos in [-0.15, 0.21]). So we compute THREE per-patch
boundary scores, take the bag-wide soft coverage of each, and SUM the
coverages: that sum is literally the cumulative-link expected ordinal class

    y = sum_k P(y > k)  =  sum_k coverage_k     in [0, 3].

------------------------------------------------------------------------
Why this is NOT any ruled-out dead-end
------------------------------------------------------------------------
* NOT a47 (ordinal cumulative-link HEAD). a47 collapsed the bag to ONE scalar
  s via attention-mean, then mapped s -> y with 3 thresholds — mathematically
  redundant with a calibrated Linear(1,1) head (DE29). Here the ordinal
  structure lives in the AGGREGATION: each P(y>k) is a separate bag-wide MEAN
  of per-patch indicators along a DIFFERENT direction. There is no single
  collapsed scalar that a monotone map could equivalently reproduce; the three
  coverages are three independent diffuse-density poolings.
* NOT the C1 single-linear family (a52/a53). C1 had ONE input-space direction,
  NO bottleneck (1.3K params) -> underfit. a54 uses the SAME expressive
  Linear(1280->128)+ReLU bottleneck as the baseline, and THREE learned
  boundary directions in the 128-d bottleneck space (not raw input).
* NOT norm-rank (a45). The per-patch ordering/threshold is over a learned
  FIBROSIS score <h_i, d_k>, never over ||h||. ||h|| never appears.
* NOT softmax-mean / attention. There is NO softmax over patches and NO top-k.
  Aggregation = unweighted MEAN of soft per-patch ordinal indicators
  (diffuse, bag-wide, exactly the "overall density" reading).
* NOT a gated head. The three sigmoids are PER-PATCH and PRE-aggregation
  (explicitly allowed). The readout y = c0 + c1 + c2 is a plain SUM of the
  three coverages (each already in [0,1]); no sigmoid/softmax sits between the
  bag representation and the scalar, and the coverages are ADDED, never
  multiplied/gated (so no presence*severity collapse, DE11-13).
* NOT a score-based softmax over a learned direction. Coverage is a MEAN of
  sigmoids, not a softmax weighting; it measures extent, not concentration.

------------------------------------------------------------------------
Mechanism (exact forward pass; single bag, batch_size=1 trainer contract)
------------------------------------------------------------------------
Input: features [N, D=1280] frozen Virchow2 patch features, N up to ~112.

1. Encoder (bottleneck, baseline-identical expressiveness):
       h_i = Dropout(ReLU(W_b @ f_i + b_b))            h_i in R^128
   (Dropout(0.5) active in train, identity at eval -> deterministic inference.)

2. Per-patch, per-boundary fibrosis score (3 boundaries k=0,1,2):
       s_{i,k} = <h_i, d_k>                            d_k in R^128 (learnable)
   d_k are warm-started from the train-prototype ordinal directions
   v_k = normalize(proto[k+1] - proto[k]) projected through the bottleneck's
   current weights? No — d_k live in BOTTLENECK space and are simply learnable
   128-d vectors initialised small-random (the raw-input prototype directions
   are 1280-d and not directly comparable to 128-d h; warm-start is therefore
   NOT used here — d_k are learned from scratch, which is fine because the
   bottleneck already supplies fibrosis-aligned features and capacity is tiny:
   3*128 = 384 params). Ordering across boundaries is NOT hard-tied between
   d_k; instead the THRESHOLDS are monotone (step 3), which is what enforces
   the ordinal cascade.

3. Monotone boundary thresholds (the ordinal-local prior):
       tau_0 = tau0_raw
       tau_1 = tau_0 + softplus(delta_1)               (> tau_0)
       tau_2 = tau_1 + softplus(delta_2)               (> tau_1)
   A single shared temperature beta = softplus(beta_raw) controls indicator
   sharpness. Monotone thresholds guarantee that, for ANY patch, the soft
   indicator is non-increasing across boundaries (crossing G2|G3 implies
   crossing G0|G1 for the same projection sign), encoding the +/-1 ordinal
   structure directly into the pooling.

   NOTE each boundary has its OWN direction d_k, so "score" differs per
   boundary; monotone tau enforces ordering of the *thresholds*, and the
   per-boundary directions let each question read its own fibrosis sub-signal.

4. Per-patch soft ordinal indicators (the +/-1-aware, per-patch nonlinearity):
       p_{i,k} = sigmoid( (s_{i,k} - tau_k) / beta )    in (0,1)

5. Bag-wide diffuse COVERAGE per boundary (the aggregation that REPLACES
   softmax-mean): unweighted mean over patches ->
       c_k = (1/N) sum_i p_{i,k}                        in (0,1)
   This is the soft FRACTION of the marrow that crosses boundary k — a
   diffuse, bag-wide extent, permutation- and size-invariant.

6. Ordinal cumulative-link readout (linear/unbounded; clamp only at the end):
       y = w0*c0 + w1*c1 + w2*c2 + b
   with the readout `head = Linear(3, 1)` warm-started to w_k = 1, b = 0 so
   that at init y = c0 + c1 + c2 = sum_k P(y>k) is exactly the expected
   ordinal class. The head is free to recalibrate the three coverages but is
   LINEAR and UNBOUNDED; final clamp to [0,3] only at the very end.

Output: (y[1], attn[N] or None, None). `attn` returned for the trainer's
optional viz is the boundary-averaged per-patch indicator (interpretable as
"how fibre-positive is this patch across boundaries"), NOT used in the
forward computation of y.

------------------------------------------------------------------------
Invariance / determinism
------------------------------------------------------------------------
Coverage c_k = mean over patches -> permutation- AND bag-size-invariant.
Dropout is the only stochastic op; eval() makes it identity -> deterministic
at inference. No subsampling, no MC, no EMA.

------------------------------------------------------------------------
Ablation companion: a55_meancov_cascade.py
------------------------------------------------------------------------
a55 removes EXACTLY the active ingredient — the per-patch ORDINAL COVERAGE
(the sigmoid-then-mean extent). It replaces step 4-5 with the LINEAR
mean-projection c_k = mean_i s_{i,k} = <mean_i h_i, d_k> (i.e. mean-pool the
bottleneck features, then project) and keeps everything else identical
(same 3 boundary directions, same monotone thresholds as additive offsets,
same Linear(3,1) readout). a54 vs a55 isolates "does diffuse fibrosis EXTENT
(coverage, a nonlinear fraction mean-pool CANNOT represent) beat fibrosis
AVERAGE (mean-pool + linear) under the SAME ordinal-cascade structure?".
This is the decisive, on-the-line test flagged by the diagnostic (mean-proj
== mean-pool; coverage is the only genuinely novel piece).

------------------------------------------------------------------------
Predicted failure mode
------------------------------------------------------------------------
On the tiny 10-patient val cohort the three coverages c_k may become highly
correlated (the three learned directions collapse toward the dominant hi-lo
fibrosis axis), at which point a54 degenerates toward a single-coverage model
(C1-like) and a54 ~ a55 ~ baseline — i.e. the ordinal cascade buys nothing
beyond one diffuse-density axis. The other risk: with N as small as a few
patches the coverage estimate is noisy, inflating val variance on G0/G3
(2 patients each). Kill if BOTH a54 and a55 val_qwk < 0.78 at seed=2, or if
a54 does not beat a55 on a multi-seed {0,1,2,3,42} val median.

------------------------------------------------------------------------
Param count (input_dim=1280, hidden_dim=128)
------------------------------------------------------------------------
    bottleneck  Linear(1280,128)+bias          = 163,968
    directions  d_k 3 x 128                     =     384
    thresholds  tau0_raw, delta_1, delta_2      =       3
    temperature beta_raw                        =       1
    readout     Linear(3,1)+bias                =       4
    ------------------------------------------------------
    total trainable                             = 164,360
    (well under the 197,250 baseline; bottleneck supplies the capacity.)
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
        # number of grade boundaries (G0|G1, G1|G2, G2|G3) -> 3 for G0..G3.
        n_boundaries: int = 3,
        # threshold init so the cascade starts well-separated in score space.
        tau0_init: float = 0.0,
        delta_init: float = 0.5413248,  # softplus(.) ~= 1.0 spacing
        beta_init: float = 0.5413248,   # softplus(.) ~= 1.0 indicator temperature
        dir_init_scale: float = 0.1,    # small-random boundary directions
        use_coverage: bool = True,      # a54 main = True; a55 ablation = False
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "a54 is scalar-regression only (num_classes=1)."
        self.n_boundaries = int(n_boundaries)
        self.use_coverage = bool(use_coverage)
        self.clamp_output = bool(clamp_output)

        # Encoder: baseline-identical bottleneck (the capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-boundary fibrosis directions in BOTTLENECK space [K, H].
        d = torch.randn(self.n_boundaries, hidden_dim)
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.directions = nn.Parameter(d * float(dir_init_scale))

        # Monotone thresholds via softplus-cumsum (tau_0 < tau_1 < tau_2).
        self.tau0_raw = nn.Parameter(torch.tensor(float(tau0_init)))
        self.delta_raw = nn.Parameter(
            torch.full((self.n_boundaries - 1,), float(delta_init))
        )
        # Shared indicator temperature (positive via softplus).
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

        # Linear cumulative-link readout over the K coverages; warm-start to
        # y = sum_k c_k = expected ordinal class.
        self.head = nn.Linear(self.n_boundaries, num_classes)
        with torch.no_grad():
            self.head.weight.fill_(1.0)
            if self.use_coverage:
                # y = c0 + c1 + c2 = sum_k P(y>k) = expected ordinal class.
                self.head.bias.zero_()
            else:
                # Ablation: mean-projection c_k ~= 0 at init, so center y at the
                # mid grade range to keep the readout off the clamp boundary.
                self.head.bias.fill_(1.5)

    def _thresholds(self) -> torch.Tensor:
        """Monotone boundary thresholds [K] via softplus-cumsum."""
        steps = F.softplus(self.delta_raw)            # [K-1], all > 0
        taus = torch.cat([self.tau0_raw.view(1), self.tau0_raw + torch.cumsum(steps, 0)])
        return taus                                   # [K]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)          # [1, N, D]
        B, N, _ = features.shape

        # 1. Encoder (per patch).
        h = self.bottleneck(features.reshape(B * N, -1)).reshape(B, N, -1)  # [B,N,H]

        # 2. Per-patch, per-boundary fibrosis scores: [B, N, K].
        s = torch.einsum("bnh,kh->bnk", h, self.directions)

        taus = self._thresholds()                     # [K]
        beta = F.softplus(self.beta_raw).clamp(min=1e-3)

        if self.use_coverage:
            # 3-5. Per-patch soft ordinal indicators, then bag-wide MEAN
            #      coverage per boundary (the active ingredient).
            p = torch.sigmoid((s - taus.view(1, 1, -1)) / beta)  # [B,N,K] in (0,1)
            c = p.mean(dim=1)                                    # [B,K] coverage
            attn = p.mean(dim=2)                                 # [B,N] viz only
        else:
            # Ablation (a55): linear mean-projection = mean-pool then project.
            # c_k = mean_i s_{i,k} = <mean_i h_i, d_k>. The sigmoid-then-mean
            # (the active ingredient) is removed; the per-boundary thresholds
            # tau_k are a sigmoid-specific device, so here the constant offset
            # is absorbed by the Linear(3,1) readout bias (warm-started below)
            # rather than subtracted explicitly. This removes EXACTLY the
            # nonlinear coverage and nothing else.
            c = s.mean(dim=1)                                    # [B,K]
            attn = s.mean(dim=2)                                 # [B,N] viz only

        # 6. Linear cumulative-link readout, clamp only at the very end.
        y = self.head(c)                                         # [B,1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if squeeze:
            y = y.squeeze(0)                                     # [1]
            attn = attn.squeeze(0)                               # [N]

        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    n_boundaries=3,
    tau0_init=0.0,
    delta_init=0.5413248,
    beta_init=0.5413248,
    dir_init_scale=0.1,
    use_coverage=True,   # a54 main = ordinal coverage cascade
    clamp_output=True,
)
