"""a66 (= a64_fibrosis_moment_field) — Distributional-moment readout of the
warm-started fibrosis projection (diffuse-density SHAPE statistics along the
prototype axis).

This is the MAIN novelty module for the a64_fibrosis_moment_field design,
delivered as a self-contained drop-in at this path. (An earlier copy lives at
a64_fibrosis_moment_field.py; this file is the screen-ready deliverable and its
ablation companion is a67_a64_fibrosis_moment_field_ablation.py.)

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
The mean projection (a53 == mean-pool) throws this shape away. The genuinely new
ingredient is the SECOND and THIRD central moments of the 1-D projection along
the warm-started fibrosis axis — diffuseness encoded as a distribution-spread
STATISTIC, orthogonal to the mean that every prior pooler reduced to.

Mechanism (exact forward-pass math; single bag, features f in R^{N x 1280})
---------------------------------------------------------------------------
  v_hat                                   # warm-started fibrosis axis (unit, learnable Parameter)
  s_i = <f_i, v_hat>                      # [N] per-patch fibrosis projection (1-D, LINEAR)
  z_i = (s_i - C) / S                     # standardise by FROZEN prototype-grade scale (C,S)
  m1  = mean_i z_i                        # 1st moment  (mean density   — the mean-pool signal)
  d_i = z_i - m1
  m2  = mean_i d_i^2                      # 2nd central moment (VARIANCE — diffuse vs focal)
  std = sqrt(m2 + eps)
  m3  = mean_i d_i^3 / (std^3 + eps)      # 3rd standardised moment (SKEW — focal-tail asymmetry)
  u   = [m1, log(1 + m2), m3]             # 3-vector shape descriptor (log on m2 for scale stability)
  y   = clamp( Linear(3 -> 1)(u) , 0, 3 ) # tiny linear readout

C and S are the mean and std of the four per-grade prototype projections
{<c_g, v_hat>}_{g=0..3} computed from the train-patient prototypes (no test
leakage; verified C=-1.287, S=7.170, standardised grade means
-1.36/-0.38/0.37/1.37). They put s_i on a grade-comparable scale so the moments
are calibrated; they are NOT learned (recomputed from the buffered prototypes,
detached, so the scale tracks v_hat if it moves but is never a per-seed DOF).

Why the recurring "collapses to mean-pool linear ablation" failure is
STRUCTURALLY IMPOSSIBLE here
----------------------------------------------------------------------
The per-patch map is a LINEAR projection s_i = <f_i, v_hat>; the only
nonlinearity (squaring/cubing) is applied AFTER centring by the BAG mean m1.
Centring-then-squaring is the one operation a per-patch linear map followed by
mean CANNOT absorb (mean o linear = linear o mean): m2 and m3 are not linear
functionals of the mean-pooled feature, so they are not expressible as
Linear(mean_i f_i). The ablation that drops m2,m3 lands EXACTLY on a53/mean-pool
along the same axis — so a66 vs a67 isolates "do higher moments of the fibrosis
projection beat its mean?" with no hidden capacity confound (both share v_hat
and the same projection, and the head init starts a66 EXACTLY at the a67
readout).

Why this is NOT a refuted family
--------------------------------
- NOT coverage / extent / threshold / fraction (a52/a54/a55): nothing counts a
  fraction above tau; we read the VALUE-distribution moments.
- NOT mean-pool / a53 / a57 (LINEAR): m2,m3 are second/third-order in the centred
  projection; not reconstructable from m1.
- NOT a free attention scorer / gated attention (baseline, a25, DE22): no
  per-patch weight at all; every patch contributes symmetrically.
- NOT ||h||-norm weighting (DE15, a45): s_i is a signed inner product with the
  fibrosis direction; ||f_i|| is never computed or used.
- NOT a per-patch NONLINEAR severity MLP pooled to its mean (a56-a63): per-patch
  map is a single LINEAR projection; the nonlinearity lives only in the
  BAG-LEVEL moment computation (post-centring).
- NOT a sigma/softmax GATE between bag rep and y (DE11-13): the readout is a
  single unbounded Linear on the 3-moment vector; clamp only at the very end.

Seed-robustness argument
-------------------------
1. Warm-start anchor: v_hat starts at the prototype `axis` (+0.84 held-out
   coverage Spearman; monotone per-grade projections -11.0/-4.0/1.4/8.5). The
   optimiser begins grade-aligned and only refines one 1280-d direction.
2. Minimal capacity: trainable = v_hat (1280) + Linear(3->1) (4) = 1284, ~150x
   below the 197K that overfit this cohort. No bottleneck MLP to memorise bags.
3. Frozen scale calibration: C,S come from prototype projections (detached), not
   learned per seed.
4. Permutation-invariant (moments are symmetric means over patches) and
   bag-size-invariant (moments normalise by N; bag duplication leaves m1,m2,m3
   and y unchanged). Deterministic at inference. Degenerate N=1 -> m2=m3=0 (safe;
   reduces to the mean term).

Ablation companion: a67_a64_fibrosis_moment_field_ablation.py — flips
`moment_mode='mean'`, masking m2,m3 to 0 so the readout is Linear on [m1,0,0]
== clamp(a*m1+b) == a53 / mean-pool along the SAME axis. Removes EXACTLY the
active ingredient (the higher moments) with everything else held fixed.

Kill criterion: abandon if a66 val_qwk < 0.78 at seed=2 AND a66 <= a67. DoD =
multi-seed audit {0,1,2,3,42}, not seed=2 alone.

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
    """Return (axis[D] unit, protos[4,D]) from the train-patient prototype cache.

    Defensive: if the cache is missing, fall back to a deterministic random axis
    and synthetic monotone prototypes so the module still instantiates and the
    interface is sane (warm-start benefit is lost, but the trainer never breaks).
    """
    if not path.is_file():
        g = torch.Generator().manual_seed(2)
        axis = torch.randn(input_dim, generator=g)
        axis = axis / axis.norm().clamp(min=1e-8)
        # Synthetic monotone per-grade prototypes along the axis so C,S are finite.
        protos = torch.stack(
            [axis * float(scale) for scale in (-11.0, -4.0, 1.4, 8.5)], dim=0
        )
        return axis, protos
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
        moment_mode: str = "moments",   # "moments" (a66 main) | "mean" (a67 ablation)
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

        # Tiny linear readout over [m1, log(1+m2), m3]. Init: weight=[1,0,0],
        # bias=1.5 so the model STARTS exactly at the mean-pool ablation (m2,m3
        # carry zero weight) and must EARN the higher moments — no capacity
        # confound.
        self.head = nn.Linear(3, num_classes)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0   # weight on m1 (standardised mean)
            self.head.bias.fill_(1.5)      # interior of [0,3]

    def _grade_scale(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """C,S = mean/std of the 4 per-grade prototype projections onto v.

        Computed from FROZEN prototypes and detached, so the standardisation
        tracks v_hat if it moves but stays a calibration constant, not a learned
        degree of freedom.
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
        else:  # "mean" ablation: zero the higher moments -> == a53 / mean-pool
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
    moment_mode="moments",   # a66 main = variance + skew of the fibrosis projection
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    eps=1e-4,
    clamp_output=True,
)
