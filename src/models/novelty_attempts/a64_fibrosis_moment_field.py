"""a64 — Distributional-moment readout of the warm-started fibrosis projection
(diffuse-density SHAPE statistics along the prototype axis).

LENS
----
Prototype-similarity field, read as a *diffuse aggregation of grade-evidence*:
project every patch onto the grade-prototype axis to get a per-patch fibrosis
score s_i (a soft assignment along the G0..G3 ordinal direction), then summarise
the *whole within-bag distribution* of {s_i} with its low-order central moments
— NOT a gated attention (no learned scorer concentrates on a few patches) and
NOT a coverage count (no threshold / fraction-of-positives).

Hypothesis (Hn_fibrosis_moment_field)
--------------------------------------
Reticulin grade = OVERALL / DIFFUSE density of the fibre meshwork. Two bags can
share the SAME mean density yet differ in grade:
  * a uniformly-moderate bag (truly diffuse fibrosis, high grade) has high mean
    AND low variance of s_i;
  * a clean-marrow bag with a few hot tiles (focal, low grade) has the SAME mean
    but high variance and positive skew.
The mean projection (a53 ≡ mean-pool) throws this shape away. The genuinely new
ingredient is the SECOND and THIRD central moments of the 1-D projection along
the warm-started fibrosis axis — diffuseness encoded as a distribution-spread
STATISTIC, exactly what the grading principle asks for, and orthogonal to the
mean that every prior pooler reduced to.

Mechanism (exact forward-pass math; single bag, features f ∈ R^{N×1280})
-------------------------------------------------------------------------
  v_hat                                   # warm-started fibrosis axis (unit, learnable Parameter)
  s_i = <f_i, v_hat>                      # [N] per-patch fibrosis projection (1-D, LINEAR)
  z_i = (s_i - C) / S                     # standardise by FROZEN prototype-grade scale (C,S)
  m1  = mean_i z_i                        # 1st moment  (mean density   — the mean-pool signal)
  d_i = z_i - m1
  m2  = mean_i d_i^2                       # 2nd central moment (VARIANCE — diffuse vs focal)
  std = sqrt(m2 + eps)
  m3  = mean_i d_i^3 / (std^3 + eps)       # 3rd standardised moment (SKEW — focal-tail asymmetry)
  u   = [m1, log(1 + m2), m3]              # 3-vector shape descriptor (log on m2 for scale stability)
  y   = clamp( Linear(3 -> 1)(u) , 0, 3 )  # tiny linear readout

C and S are the mean and std of the four per-grade prototype projections
{<c_g, v_hat>}_{g=0..3} computed ONCE from train-patient prototypes (no test
leakage). They put s_i on a grade-comparable scale so the moments are
calibrated; they are NOT learned (recomputed from the buffered prototypes only
to track v_hat if it moves, see `_grade_scale`). This is a deliberate fixed
population statistic, not a per-seed degree of freedom.

Why the recurring "collapses to mean-pool linear ablation" failure is
STRUCTURALLY IMPOSSIBLE here
----------------------------------------------------------------------
The historical collapse (a56<a57, a59/a62 → mean) happened because a NONLINEAR
PER-PATCH map followed by an (un)weighted MEAN can be matched or beaten by its
LINEAR per-patch ablation, since mean ∘ linear = linear ∘ mean. Here the
per-patch map is a FROZEN-then-fine-tuned LINEAR projection s_i = <f_i, v_hat>,
and the only nonlinearity (squaring/cubing) is applied AFTER centring by the
BAG MEAN m1. Centring-then-squaring is the one operation a per-patch linear map
followed by mean CANNOT absorb: m2 and m3 are not linear functionals of the
mean-pooled feature, so they are not expressible as Linear(mean_i f_i). The
ablation that drops m2,m3 lands EXACTLY on a53/mean-pool — so a64 vs a64-mean
isolates "do higher moments of the fibrosis projection beat its mean?" with no
hidden capacity confound (both share v_hat and the same projection).

Why this is NOT a refuted family
--------------------------------
- NOT coverage / extent / threshold / fraction (a52/a54/a55, refuted 3×):
  nothing counts "fraction of patches above τ"; we read the VALUE-distribution
  moments, not a count of positives.
- NOT mean-pool / a53 / a57 (LINEAR): m2,m3 are second/third-order in the
  centred projection; they cannot be reconstructed from m1.
- NOT a free attention scorer / gated attention (baseline, a25, DE22): there is
  no per-patch weight at all — every patch contributes symmetrically to the
  moments. No softmax, no concentration.
- NOT ||h||-norm weighting (DE15, a45, refuted): the projection s_i is a signed
  inner product with the fibrosis direction; ||f_i|| is never computed or used.
- NOT a per-patch NONLINEAR severity MLP pooled to its mean (a56–a63, collapsed):
  the per-patch map is a single LINEAR projection; the nonlinearity lives only
  in the BAG-LEVEL moment computation (post-centring), where it provably cannot
  be absorbed into a per-patch linear map.
- NOT a σ/softmax GATE between bag rep and ŷ (DE11–13): the readout is a single
  unbounded Linear on the 3-moment vector; clamp only at the very end.

Seed-robustness argument
-------------------------
1. Warm-start anchor. v_hat is initialised at the prototype `axis` (the +0.84
   held-out-coverage fibrosis direction; per-grade projections −11.0/−4.0/1.4/8.5
   are monotone and well separated). The optimiser starts grade-aligned and only
   refines a single 1280-d direction; the small 214-ROI val cohort cannot drive
   it into a non-generalising corner (the a40/baseline seed-fragility root cause,
   and the a49/a50 from-scratch-direction collapse). Ablation `warm_start=False`
   re-tests whether the grounded init is what buys robustness.
2. Minimal capacity. Trainable = v_hat (1280) + Linear(3→1) (4) = 1284 params,
   ~150× below the 197K that overfit this cohort and below a53's regime. No
   bottleneck MLP to memorise the train bags. The three moments are population
   shape statistics, low-variance across resamples of the bag.
3. Scale calibration is FROZEN. C,S come from prototype projections, not learned
   per seed — removes a degree of freedom that could chase val noise.
4. Permutation-invariant (all moments are symmetric means over patches) and
   bag-size-invariant (moments normalise by N; duplicating the bag leaves
   m1,m2,m3 and hence ŷ unchanged). Deterministic at inference (no sampling).
   Degenerate N=1 → m2=m3=0 (safe; reduces to the mean term).

Ablation companion: a65_fibrosis_moment_field_meanonly.py — flips
`moment_mode='mean'`, which masks m2,m3 to 0 so the readout is Linear on [m1,0,0]
≡ clamp(a·m1+b) ≡ a53 / mean-pool along the SAME axis. Removes EXACTLY the active
ingredient (the higher moments) with everything else (warm-start, scale, head)
held fixed.

Kill criterion: abandon if a64 val_qwk < 0.78 at seed=2 AND a64 ≤ a65
(higher moments add nothing over the mean). DoD = multi-seed audit {0,1,2,3,42},
not seed=2 alone.

Param count (input_dim=1280): v_hat 1280 + Linear(3,1) weight 3 + bias 1 = 1284.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_prototypes(path: Path, input_dim: int):
    """Return (axis[D] unit, protos[4,D]) from the train-patient prototype cache."""
    assert path.is_file(), (
        f"Prototype cache not found: {path}\n"
        f"Run `python scripts/compute_grade_prototypes.py` to generate it."
    )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    axis = blob["axis"].float()
    axis = axis / axis.norm().clamp(min=1e-8)
    assert axis.shape[0] == input_dim, f"axis dim {axis.shape[0]} != input_dim {input_dim}"
    p = blob["prototypes"]
    protos = torch.stack([p[g].float() for g in range(4)], dim=0)  # [4, D]
    return axis, protos


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        moment_mode: str = "moments",   # "moments" (a64 main) | "mean" (a65 ablation)
        warm_start: bool = True,        # init v_hat at the prototype fibrosis axis
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False
        eps: float = 1e-4,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert moment_mode in ("moments", "mean"), moment_mode
        self.moment_mode = moment_mode
        self.eps = float(eps)
        self.clamp_output = clamp_output

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        axis, protos = _load_prototypes(path, input_dim)

        if warm_start:
            v0 = axis.clone()
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v0 = torch.randn(input_dim, generator=g)
            v0 = v0 / v0.norm().clamp(min=1e-8)

        # Learnable fibrosis direction (unit warm-start; raw input space).
        self.v = nn.Parameter(v0)
        # Frozen per-grade prototype vectors (for scale calibration C,S). Buffer
        # so they move with the module device but receive no gradient.
        self.register_buffer("prototypes", protos)  # [4, D]

        # Tiny linear readout over [m1, log(1+m2), m3]. Init: lean on m1, scaled
        # so a standardised mean of ~+1.5 (≈ G1/G2 region) maps near mid-range,
        # leaving the optimiser to calibrate. m2,m3 start with zero weight so the
        # model begins AT the mean-pool ablation and must EARN the higher moments.
        self.head = nn.Linear(3, num_classes)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0   # weight on m1 (standardised mean)
            self.head.bias.fill_(1.5)      # interior of [0,3]

    def _grade_scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """C,S = mean/std of the 4 per-grade prototype projections onto v.

        Computed from FROZEN prototypes (no grad through them shapes the scale,
        but v_hat does flow so the standardisation tracks the direction if it
        moves). Detached to keep it a calibration constant, not a learned DOF.
        """
        with torch.no_grad():
            proj = self.prototypes @ self.v.detach()   # [4]
            C = proj.mean()
            S = proj.std(unbiased=False).clamp(min=1e-3)
        return C, S

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        s = features @ self.v                       # [N] per-patch fibrosis projection
        C, S = self._grade_scale()
        z = (s - C) / S                             # [N] grade-calibrated projection

        m1 = z.mean()                               # 1st moment (mean density)

        if self.moment_mode == "moments":
            d = z - m1                              # centre by BAG mean
            m2 = (d * d).mean()                     # variance (diffuse vs focal)
            std = (m2 + self.eps).sqrt()
            m3 = (d.pow(3)).mean() / (std.pow(3) + self.eps)  # standardised skew
            u = torch.stack([m1, torch.log1p(m2), m3]).view(1, 3)
        else:  # "mean" ablation: zero the higher moments → ≡ a53 / mean-pool
            zero = torch.zeros((), device=z.device, dtype=z.dtype)
            u = torch.stack([m1, zero, zero]).view(1, 3)

        y = self.head(u)                            # [1, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, s.detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    moment_mode="moments",   # a64 main = variance + skew of the fibrosis projection
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    eps=1e-4,
    clamp_output=True,
)
