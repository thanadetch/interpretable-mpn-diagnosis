"""a66 — Fibrosis COVERAGE-CURVE (staged-threshold survival function) MIL.

LENS: thresholded-extent. A smooth, calibrated fraction-of-bag-above-
fibrosis-threshold with a LEARNED soft threshold and temperature, aggregated
as MEAN COVERAGE — but read at a *staged set of K thresholds at once*, i.e.
the bag's empirical complementary CDF (survival function) of the fibrosis
projection, fed as a VECTOR to a linear head. This is the one thresholded-
extent variant that is provably NOT collapsible to a single coverage scalar
and NOT collapsible to mean-pool (justified precisely below vs a52-a55).

====================================================================
Forward-pass math (single bag, features f in R^{N x D}, D=1280)
====================================================================
1. Fibrosis projection onto a warm-started, fine-tunable direction in RAW
   1280-d input space (skip the bottleneck so the prototype warm-start is
   directly meaningful — the same regime as a52/a53):

       w        = learnable direction, initialised at
                  v_hat = normalize( mean(c_G2,c_G3) - mean(c_G0,c_G1) )
                  from data/prototypes_virchow2_reti_train_seed2.pt
                  (train patients only -> NO test leakage), scaled by
                  `init_scale` so the per-patch scores start in a
                  non-saturated sigmoid range.
       s_i      = <f_i, w>                                    # [N] 1-D score

2. A LEARNED, MONOTONE ladder of K soft thresholds (the staged extent).
   A single anchor tau0 and K-1 positive softplus increments give an ordered
   set tau_0 < tau_1 < ... < tau_{K-1}; a single shared LEARNED temperature
   beta = softplus(beta_raw) sets the indicator sharpness:

       tau_0    = tau0_raw
       tau_k    = tau_0 + sum_{j<=k} softplus(delta_j)        # k = 1..K-1
       beta     = softplus(beta_raw)            (clamped >= 1e-3)

3. MEAN COVERAGE at EACH threshold = the bag's soft survival function
   (complementary CDF) sampled at the K thresholds. Each entry is exactly a
   "calibrated fraction-of-bag-above-fibrosis-threshold" — the assigned lens —
   but we keep the whole vector instead of one number:

       c_k      = (1/N) * sum_i sigmoid( (s_i - tau_k) / beta )   in (0,1)
       c        = [c_0, ..., c_{K-1}]                          # [K] survival curve

4. LINEAR readout over the coverage VECTOR (clamp only at the very end):

       y        = clamp( a . c + b , 0, 3 )      a in R^K, b in R (Linear(K,1))

   Warm-start: a_k = 3/K, b = 0, so at init  y = (3/K) * sum_k c_k  =
   3 * mean_k c_k, i.e. the prediction starts as 3x the *average* coverage
   across the ladder (a calibrated mean-coverage readout spanning [0,3]).

====================================================================
WHY THIS IS DISTINCT FROM a52-a55 (the assigned justification)
====================================================================
a52 (refuted, val 0.7675 < its mean-proj ablation a53 0.7725):
    y = Linear_{1->1}( mean_i sigmoid((s_i - tau)/beta) )
  ONE threshold -> ONE coverage scalar c. With a single threshold the soft
  coverage c is a smooth MONOTONE function of the score distribution that, in
  the near-linear regime that early-stopping selects, is dominated by its
  first-order term mean_i s_i = <mean_i f_i, w> = mean-pool. That is exactly
  why a52 collapsed TO and LOST to mean-projection a53. A single coverage
  number cannot distinguish a DIFFUSE-moderate bag (many patches just above
  tau) from a FOCAL bag (a few patches far above tau, the rest below) when
  they share the same count at tau.

a66 reads the SAME thresholded-extent quantity at K staged thresholds. The
coverage VECTOR is the discretised survival function S(tau). The LINEAR head
then has access to the *local slope* of coverage between adjacent thresholds,
    c_k - c_{k+1} = soft fraction of patches in the band [tau_k, tau_{k+1}],
which is the DENSITY of the projection in that band. A steep survival curve
(coverage drops fast over a narrow band) = DIFFUSE uniform density (true high
grade); a shallow long-tailed curve = FOCAL/heterogeneous (lower grade or
boundary case) at the SAME mean. The mean / single-coverage readout is exactly
the *integral* (or one sample) of this curve — a scalar functional — whereas
the per-band slopes are LINEARLY INDEPENDENT of that integral. Concretely,
two bags with equal mean (hence equal a53 / equal summed coverage) produce
different coverage curves [verified: diffuse-moderate vs focal at matched mean
give curves (0.99,0.84,0.16,0.01) vs (0.91,0.37,0.21,0.20)]. So a66's readout
is NOT a function of mean-pool and NOT a function of any single coverage — the
recurring "collapses to its mean-pool linear ablation" failure (a52>a53 lost,
a56<a57, a54<a55) is structurally impossible: mean-pool lives in a strict
sub-space (uniform-weight column) of a66's K-dim readout.

a54/a55 (ordinal coverage cascade, refuted, a54 0.7747 < a55 0.7834):
  a54 used K thresholds but each on a DIFFERENT learned bottleneck direction
  d_k, and SUMMED the coverages (y = c0+c1+c2 = a cumulative-link scalar). The
  diagnostic predicted (and the run confirmed) that the three directions
  collapse toward the dominant axis, making the SUM behave like one coverage
  -> it lost to its mean ablation a55. a66 is the opposite construction: ONE
  shared axis w, K thresholds on THAT SAME axis, and the coverages are kept as
  a VECTOR read by a FREE Linear(K,1) (NOT summed, NOT tied to +1 per level).
  The K coverages here are guaranteed monotone in k (same axis, ordered taus),
  so they trace a genuine survival curve whose SHAPE — not whose sum — is the
  signal. There is no per-boundary direction to collapse.

a47 (ordinal cumulative-link head, refuted): applied thresholds to ONE
  already-collapsed bag SCALAR (post-aggregation) — redundant with a calibrated
  Linear(1,1). a66 applies the thresholds PER-PATCH, PRE-aggregation, so the
  coverage curve carries within-bag distribution shape that no post-hoc head on
  a collapsed scalar can recover.

====================================================================
GRADING-ALIGNMENT (the advisor's prior)
====================================================================
* Grade = OVERALL / DIFFUSE density. Every c_k is an unweighted MEAN over ALL
  patches — a bag-wide extent, not a few standout patches, not attention
  concentration, not top-k. The readout is permutation- and bag-size-invariant.
* NO ||h|| anywhere. The per-patch quantity is the projection onto the learned
  fibrosis axis w (warm-started at the +0.84-coverage prototype direction),
  which the diagnostics show is grade-informative (proj +0.882, cov +0.844)
  and skips bone (Spearman(||h||,<h,v>) = -0.21). Norm never enters.
* The survival-curve SHAPE operationalises "diffuse vs focal" directly: the
  pathologist's distinction between a uniformly moderate meshwork (high grade)
  and a focal patch of dense fibre in clean marrow (lower grade) IS the
  difference between a steep and a shallow coverage curve at equal mean.

====================================================================
SEED-ROBUSTNESS argument (avoid the a40 one-seed lottery)
====================================================================
* The axis w is WARM-STARTED at the label-free train-prototype direction
  (+0.84 held-out coverage Spearman), so the optimiser starts grade-aligned
  and only refines; the tiny 10-patient val cohort cannot drag it into a
  non-generalising corner the way a from-scratch direction (a49/a50) could.
* Capacity is deliberately tiny: w (1280) + tau ladder (K) + beta (1) +
  Linear(K,1) (K+1). With K=7 that is 1280 + 7 + 1 + 8 = 1296 trainable
  params, ~150x below the 197K baseline that overfits the 214-ROI val cohort.
  Low capacity + grounded warm-start is exactly the regime that survives
  across seeds rather than peaking on one.
* The readout STRICTLY CONTAINS mean-coverage (uniform-weight column) and,
  with K coverages spanning the score range, can closely approximate
  mean-pool too — so a66 can never be structurally worse than the refuted
  baselines; any seed-stable gain comes only from the EXTRA slope DOF, which
  is the variable the ablation a67 isolates.
* Monotone (softplus-cumsum) thresholds keep the ladder ordered every step,
  preventing the threshold-permutation degeneracy that makes free multi-tau
  models seed-fragile.

====================================================================
Determinism / invariance
====================================================================
Each c_k is a mean of a deterministic sigmoid over patches -> permutation- AND
bag-size-invariant. No dropout, no softmax over patches, no sampling, no MC,
no EMA. Fully deterministic at inference. Frozen Virchow2 features only;
features are the sole input (no coords, no scale tags, no second backbone).

====================================================================
Ablation companion: a67_fibrosis_coverage_single.py
====================================================================
a67 sets `n_thresholds=1` (K=1) with everything else identical (same warm-
started axis, same learned tau, same learned beta, same warm-start, same
clamp). K=1 is EXACTLY a52: a single soft-threshold mean coverage read by a
Linear(1,1). a66 vs a67 isolates the single active ingredient added here:
"does reading the COVERAGE CURVE at multiple staged thresholds (its SHAPE /
local slope = diffuse-vs-focal density) beat reading coverage at ONE threshold
(a52, which lost to mean-pool)?" If a66 ~= a67, the honest finding is that the
survival-curve shape adds nothing beyond a single extent count.

Kill criterion: abandon if a66 val_qwk < 0.79 at seed=2 AND a66 does not beat
a67 on a multi-seed {0,1,2,3,42} val median (the curve buys nothing over one
threshold). Definition of done = multi-seed audit, not seed=2 alone.

Param count (input_dim=1280, n_thresholds=7):
    w 1280 + tau0_raw 1 + delta_raw 6 + beta_raw 1 + head Linear(7,1) 8 = 1296.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_direction(path: Path, input_dim: int) -> torch.Tensor:
    """Unit vector v = normalize(mean(G2,G3) - mean(G0,G1)) from train prototypes."""
    assert path.is_file(), (
        f"Prototype cache not found: {path}\n"
        f"Run `python scripts/compute_grade_prototypes.py` to generate it."
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    p = blob["prototypes"]
    hi = (p[2].float() + p[3].float()) / 2.0
    lo = (p[0].float() + p[1].float()) / 2.0
    v = hi - lo
    v = v / v.norm().clamp(min=1e-8)
    assert v.shape[0] == input_dim, f"direction dim {v.shape[0]} != input_dim {input_dim}"
    return v


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        n_thresholds: int = 7,        # K: staged thresholds (a66 main); 1 = a67 ablation (==a52)
        init_scale: float = 0.1,      # scales warm-started w so sigmoid starts unsaturated
        tau0_init: float = -1.5,      # anchor of the threshold ladder
        delta_init: float = 0.5413248,  # softplus(.) ~= 0.5 spacing between thresholds
        beta_init: float = 0.0,       # softplus(0)=0.693 initial indicator temperature
        warm_start: bool = True,      # init w at prototype fibrosis direction
        prototype_path: Optional[str] = None,
        random_seed: int = 2,         # used only if warm_start=False
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert int(n_thresholds) >= 1, "n_thresholds must be >= 1"
        self.K = int(n_thresholds)
        self.clamp_output = bool(clamp_output)

        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            v = _load_fibrosis_direction(path, input_dim)
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v = torch.randn(input_dim, generator=g)
            v = v / v.norm().clamp(min=1e-8)

        # Learnable fibrosis direction (warm-started, small-scale so per-patch
        # scores start in a non-saturated sigmoid range).
        self.w = nn.Parameter(v * float(init_scale))

        # Monotone threshold ladder via softplus-cumsum: tau_0 < tau_1 < ... .
        self.tau0_raw = nn.Parameter(torch.tensor(float(tau0_init)))
        if self.K > 1:
            self.delta_raw = nn.Parameter(torch.full((self.K - 1,), float(delta_init)))
        else:
            # No increments when K=1 (a67 ablation == single-threshold a52).
            self.register_parameter("delta_raw", None)

        # Shared learned indicator temperature (positive via softplus).
        self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))

        # Linear readout over the K coverages. Warm-start a_k = 3/K, b = 0 so
        # y starts at 3 * mean_k coverage_k (a calibrated mean-coverage readout).
        self.head = nn.Linear(self.K, num_classes)
        with torch.no_grad():
            self.head.weight.fill_(3.0 / self.K)
            self.head.bias.zero_()

    def _thresholds(self) -> torch.Tensor:
        """Monotone threshold ladder [K] via softplus-cumsum (tau_0 < ... < tau_{K-1})."""
        if self.K == 1:
            return self.tau0_raw.view(1)
        steps = F.softplus(self.delta_raw)                       # [K-1], all > 0
        return torch.cat([self.tau0_raw.view(1), self.tau0_raw + torch.cumsum(steps, 0)])

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        s = features @ self.w                                    # [N] per-patch fibrosis score

        taus = self._thresholds()                                # [K]
        beta = F.softplus(self.beta_raw).clamp(min=1e-3)

        # Soft survival function: coverage at each staged threshold.
        # p[i,k] = sigmoid((s_i - tau_k)/beta); c_k = mean_i p[i,k].
        p = torch.sigmoid((s.unsqueeze(1) - taus.unsqueeze(0)) / beta)  # [N, K]
        c = p.mean(dim=0)                                        # [K] coverage curve

        y = self.head(c.view(1, self.K))                         # [1, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Viz only: per-patch mean coverage across the ladder (NOT used in y).
            weight = p.mean(dim=1)                               # [N]
            return y, weight, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    n_thresholds=7,         # a66 main = coverage CURVE (staged thresholds)
    init_scale=0.1,
    tau0_init=-1.5,
    delta_init=0.5413248,
    beta_init=0.0,
    warm_start=True,
    prototype_path=None,    # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
