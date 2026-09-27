"""a64 — Distributional-moment readout of the fibrosis projection.

LENS (distributional): model the DISTRIBUTION of per-patch fibrosis-projection
scores {s_i = <f_i, v>} as the bag descriptor — diffuse density, NOT a few
standout patches. Read the low-order central MOMENTS (mean, spread, skew) of
that 1-D distribution and map them linearly to the grade.

------------------------------------------------------------------------------
FORWARD-PASS MATH (single bag, features f in R^{N x 1280}; v in R^{1280}):

  1. Per-patch fibrosis projection (1-D, frozen LINEAR map -> cannot collapse
     to a per-patch nonlinearity, see "why distinct" below):
         s_i = <f_i, v>                                   for i = 1..N      [N]
     v is warm-started at the train-prototype fibrosis direction
         v0 = normalize( mean(c_G2,c_G3) - mean(c_G0,c_G1) )
     (held-out coverage Spearman +0.84). By default v is a FROZEN buffer
     (freeze_axis=True) so the dominant grade signal can never drift into a
     non-generalising, seed-specific corner — the a40 lottery root cause.

  2. Bag-level central moments of {s_i} (each permutation- AND bag-size-
     invariant: every term is a mean over patches):
         m1 = (1/N) * sum_i s_i                            # mean  (density)
         d_i = s_i - m1
         m2 = (1/N) * sum_i d_i^2                          # variance (spread)
         sd = sqrt(m2 + eps)
         g1 = (1/N) * sum_i d_i^3 / (sd^3 + eps)           # skewness (shape)
     m2 and g1 are STANDARDIZED-form statistics: m2 is in raw projection
     units squared, g1 is dimensionless. To make the linear head scale-stable
     across the (frozen) axis we feed the head a scale-matched moment vector
         z = [ m1 ,  sqrt_signed(m2) ,  g1 ]
     where sqrt_signed(m2) = sign(m2)*sqrt(|m2|) = sd puts the spread in the
     SAME units as m1 (so no single coordinate dominates the Linear by scale
     alone). (sd >= 0 always; sqrt_signed kept general for the random-axis
     ablation where the warm-start sign is lost.)
         z = [ m1 , sd , g1 ]                                              [3]

  3. Linear readout + clamp (regression target range G0..G3):
         y = clamp( w_m . z + b , 0, 3 )                                 [1,1]
     Head warm-start: w_m = [+0.3, 0, 0], b = +1.5 so the model STARTS as a
     mean-pool regressor centred mid-grade (m1 spans roughly [-10,+9] across
     grades, so 0.3*m1 spans ~[-3,+3] around 1.5). The optimiser then chooses
     whether the spread/skew coordinates earn weight. (Edge case N==1: m2=0,
     g1=0 -> the bag is graded by its single projection, well-defined.)

  Returns (y[1,1], aux1, aux2) per trainer contract. aux1 = the per-patch
  projection s (offered as "attention"/diagnostic when return_attention=True),
  aux2 = None. No auxiliary loss (trainer computes SmoothL1 on y only).

------------------------------------------------------------------------------
GRADING ALIGNMENT (the pathology prior the design must respect):
  Grade = OVERALL / DIFFUSE density of the reticulin meshwork, a bag-wide
  property. m1 captures the average density (the validated mean-pool signal,
  Spearman +0.86 on all 1330 bags here). m2 (spread) and g1 (skew) read the
  SHAPE of the diffuse-density distribution: a truly diffuse high-grade bag
  has its projection mass shifted up with a thin DOWN-tail toward residual
  clean patches (NEGATIVE skew); a focal / G0-with-a-few-positive-tiles bag
  has a clean-marrow floor with an UP-tail of a few hot patches (POSITIVE
  skew) at a possibly similar mean. Skew of the projection is therefore a
  literal "diffuse vs focal" statistic. This is diffuseness as a STATISTIC of
  the whole distribution, not attention concentration on a few patches.

  Empirically grounded on all 1330 Virchow2 reti bags (this repo, no test
  leakage in the axis — v from train prototypes only):
      Spearman(grade, mean m1)  = +0.858   (dominant)
      Spearman(grade, std  sd)  = -0.127
      Spearman(grade, skew g1)  = -0.517   (monotone: G0 +0.45 ... G3 -0.47)
  After linearly residualising the mean out, std and skew each still carry
  ~+0.07 Spearman with grade -> a small but real, mean-ORTHOGONAL signal. The
  design is therefore a mean-DOMINATED readout with a shape correction, not a
  shape-only gamble.

  NO ||h|| weighting anywhere (norm is grade-uninformative, Spearman ~0 per
  results/diag/norm_vs_grade.md). The only per-patch map is the fibrosis
  projection <f_i, v>; norm is never computed or used.

------------------------------------------------------------------------------
WHY THIS IS DISTINCT FROM EVERY TRIED MECHANISM (a01-a63):
  * NOT mean-pool / a53 (= exactly m1) nor a57 (linear severity = mean-pool):
    m2 and g1 are 2nd/3rd CENTRAL moments. They are computed by centering then
    squaring/cubing AT THE BAG LEVEL, AFTER aggregation. This is the one place
    a nonlinearity CANNOT be absorbed into a per-patch linear map, so the
    recurring "nonlinear-per-patch collapses to its mean-pool linear ablation"
    failure (a56->a57, a58-a63) is STRUCTURALLY IMPOSSIBLE here: m2,g1 are not
    functions of mean_i f_i and cannot be reconstructed from m1.
  * NOT coverage / a52 / a54 / a55: those reduce the distribution to a single
    soft-threshold COUNT (a fraction). No sigmoid, no threshold, no fraction
    here — we read the moments (mean, spread, shape) of the raw projection.
  * NOT a soft-quantile / order statistic (the audit's companion gap): no rank,
    no quantile level, no L-estimator — central moments are a different (and
    cheaper, O(N)) read of the same distribution.
  * NOT a sigmoid GATE between bag-rep and prediction (DE11-13 gradient
    bottleneck): the head is a single unbounded Linear(3->1) then clamp; no
    sigmoid anywhere on the path to y.
  * NOT prototype-as-attention (DE22 / a25/a26): v is used to PROJECT, then we
    read the distribution's moments; it is never a softmax attention scorer.
  * NOT a learned-from-random direction in a bottleneck (DE32 / a49/a50): the
    projection lives in RAW 1280-d input space and is FROZEN at the warm-start
    by default, so the +0.84 axis is directly meaningful and seed-stable.

------------------------------------------------------------------------------
ABLATION COMPANION: a65_fibrosis_projection_mean_only.py
  Sets moment_mode='mean': feeds the head z = [m1, 0, 0] -> y = clamp(w*m1+b)
  = mean-pool + linear head (== a53). This removes EXACTLY the active
  ingredient (the higher moments m2,g1). a64 vs a65 isolates the one question:
  "do the SPREAD and SKEW of the fibrosis projection beat its MEAN?".
  (A second, axis-robustness ablation is exposed via warm_start=False, which
  re-initialises v from random noise instead of the prototype axis, isolating
  "does the grounded warm-start beat a from-scratch direction across seeds?".)

------------------------------------------------------------------------------
SEED-ROBUSTNESS ARGUMENT:
  (1) The dominant coordinate m1 rides a FROZEN, label-grounded axis (+0.84
      held-out, +0.86 here) — it cannot wander per seed, unlike learned-from-
      random directions (a40/a49/a50 lottery).
  (2) Trainable params ~= 5 (head w[3] + b[1], axis frozen) -> ~3-4 orders of
      magnitude below the 197K regime that overfit the 214-ROI val cohort.
      With 5 params there is almost no capacity to fit val noise.
  (3) The higher-moment coordinates enter ADDITIVELY through a linear head
      warm-started at zero weight, so at worst the optimiser keeps them off and
      a64 degrades gracefully to its a65 mean-pool ablation rather than to a
      degenerate corner. Improvement, if any, is a monotone shape correction
      that is consistent across seeds because the moments themselves are
      population statistics, not learned features.

CAPACITY (input_dim=1280, freeze_axis=True default):
    axis v ............. 1280 (FROZEN buffer, 0 trainable)
    head Linear(3->1) .. 3 + 1 = 4 trainable
    feat_scale ......... 1 trainable
    -----------------------------------------------------
    trainable total .... ~5     (vs 197,250 baseline)
  If freeze_axis=False (fine-tune variant), axis becomes trainable: 1285.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_direction(path: Path, input_dim: int) -> torch.Tensor:
    """Unit vector v = normalize(mean(G2,G3) - mean(G0,G1)) from train prototypes.

    Train-patients only -> no test leakage. This is the +0.84 held-out coverage
    direction used by a52/a53; here we read its distribution's moments.
    """
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
        moment_mode: str = "moments",   # "moments" (a64 main) | "mean" (a65 ablation)
        warm_start: bool = True,        # init v at prototype fibrosis axis (else random)
        freeze_axis: bool = True,       # keep v frozen (seed-robust) vs fine-tune
        feat_scale_init: float = 0.3,   # global scale on the moment vector before head
        head_w_init: float = 1.0,       # init weight on m1 (others start at 0)
        head_b_init: float = 1.5,       # mid-grade bias
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False
        eps: float = 1e-6,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert moment_mode in ("moments", "mean"), moment_mode
        self.moment_mode = moment_mode
        self.eps = float(eps)
        self.clamp_output = clamp_output

        # ---- fibrosis projection axis (raw 1280-d) ----
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            v = _load_fibrosis_direction(path, input_dim)
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v = torch.randn(input_dim, generator=g)
            v = v / v.norm().clamp(min=1e-8)

        if freeze_axis:
            self.register_buffer("v", v)            # frozen, 0 trainable params
            self._axis_is_param = False
        else:
            self.v = nn.Parameter(v)                # fine-tuned (1280 trainable)
            self._axis_is_param = True

        # ---- global scale so the head sees a tame moment vector ----
        # (one scalar; keeps z ~ O(1) regardless of the axis norm so head
        #  weights are scale-stable across seeds / warm-start vs random.)
        self.feat_scale = nn.Parameter(torch.tensor(float(feat_scale_init)))

        # ---- linear readout: z = [m1, sd, g1] -> y ----
        self.head = nn.Linear(3, num_classes)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = float(head_w_init)   # start on m1 (mean-pool)
            self.head.bias.fill_(float(head_b_init))

    def _axis(self) -> torch.Tensor:
        return self.v if self._axis_is_param else self.v

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        v = self._axis().to(features.dtype)
        s = features @ v                          # [N] per-patch fibrosis projection

        m1 = s.mean()                             # mean (diffuse density)
        if self.moment_mode == "moments" and s.numel() > 1:
            d = s - m1
            m2 = (d * d).mean()                   # variance (spread)
            sd = torch.sqrt(m2 + self.eps)        # std, same units as m1
            m3 = (d * d * d).mean()               # 3rd central moment
            g1 = m3 / (sd.pow(3) + self.eps)      # skewness (shape; diffuse<->focal)
        else:
            # "mean" ablation, or degenerate single-patch bag -> shape terms off.
            sd = torch.zeros((), dtype=s.dtype, device=s.device)
            g1 = torch.zeros((), dtype=s.dtype, device=s.device)

        # scale-matched moment vector, then a global tame-scale.
        z = torch.stack([m1, sd, g1]) * self.feat_scale          # [3]
        y = self.head(z).view(1, 1)                              # [1,1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, s, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    moment_mode="moments",   # a64 main = mean + spread + skew of the projection
    warm_start=True,
    freeze_axis=True,        # frozen, label-grounded axis -> seed-robust
    feat_scale_init=0.3,
    head_w_init=1.0,
    head_b_init=1.5,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    eps=1e-6,
    clamp_output=True,
)
