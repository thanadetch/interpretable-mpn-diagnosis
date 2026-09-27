"""a72 — Pairwise within-bag density-CONTRAST (relational / 2nd-order readout
of the warm-started fibrosis projection).

LENS
----
Project every patch onto the warm-started fibrosis axis to get a per-patch
density score s_i = <f_i, v> (LINEAR, raw 1280-d space, v trainable), then read
the WHOLE bag with its mean AND its mean ABSOLUTE PAIRWISE DIFFERENCE — a
Gini-style SPREAD that asks "is the fibrosis uniform across the marrow, or is it
a few hot tiles in clean marrow?". This is a 2nd-ORDER / RELATIONAL statistic
(every patch is compared to every OTHER patch), not a contrast of each patch to
the bag centroid (a59/a62, 1st-order) and not a coverage count (a52, threshold).

Hypothesis (Hn_fibrosis_pairwise_spread)
----------------------------------------
Reticulin grade = OVERALL / DIFFUSE density of the fibre meshwork over the whole
ROI. Two bags can share the SAME mean density yet differ in diffuseness:
  * a truly DIFFUSE high-density bag (every patch moderately fibrotic = high
    grade) has LOW pairwise contrast — all s_i are similar, so D is small;
  * a FOCAL bag (a few hot patches in otherwise clean marrow = lower grade for
    the SAME mean) has HIGH pairwise contrast — s_i are spread apart, D large.
The advisor's grading principle is "diffuse density across the whole bag", so a
LOW pairwise spread at a given mean is evidence of genuine diffuse fibrosis,
while a HIGH spread at the same mean flags focality. The mean projection (a53 ==
mean-pool) throws this away; the spread is the genuinely new ingredient.

Mechanism (exact forward-pass math; single bag, features f in R^{N x 1280})
---------------------------------------------------------------------------
  v                                       # warm-started fibrosis axis (unit, learnable Parameter)
  s_i  = <f_i, v>                         # [N] per-patch density score (1-D, LINEAR)
  mean = mean_i s_i                       # 1st-order mean density (the mean-pool signal)
  D    = mean_{i,j} phi(s_i, s_j)         # bag SPREAD scalar (2nd-order)
         phi = |s_i - s_j|  (a72 main, "spread")  -> mean-abs-pairwise-difference (Gini)
         phi = s_i * s_j    (a73 abl., "product") -> D = (mean_s)^2 (pure fn of the mean)
  y    = clamp( Linear([mean, D]) , 0, 3 )  # tiny 2-in linear readout

Efficient O(N) Gini (NO N^2 materialisation): the mean absolute pairwise
difference of a 1-D set equals a sorted-order weighted sum,
    D = (2 / N^2) * sum_k (2k - N + 1) * s_(k)      (k = 0..N-1, ascending s_(k))
which we compute from a single torch.sort — exact, permutation-invariant,
bag-size-invariant, and cheap for any N (the O(N^2) brute form is used only as a
documented fallback / cross-check, never on the hot path). This includes the
i=j diagonal (phi=0), matching the mean over ALL ordered pairs.

Why the recurring "collapses to mean-pool linear ablation" failure is
STRUCTURALLY IMPOSSIBLE for the main module
-------------------------------------------------------------------
The per-patch map is a single LINEAR projection s_i = <f_i, v>; the nonlinearity
lives ONLY in the BAG-LEVEL statistic. The mean-absolute-pairwise-difference is
NOT a function of the mean: two bags with identical mean_s can have D=0 (all
equal) or D large (spread apart). It is a genuinely 2nd-order functional of the
1-D projection that a per-patch-linear-then-mean pooler CANNOT reproduce
(mean o linear = linear o mean, but |.| of pairwise gaps is not). So a72 cannot
silently degrade to a53/mean-pool the way the a56-a63 per-patch nonlinear pools
did (where the ablation beat the main 6 times).

The ablation (a73) swaps phi -> s_i * s_j, for which
    D = mean_{i,j} s_i s_j = (mean_i s_i)^2 = mean^2,
a PURE function of the mean. So [mean, D] = [mean, mean^2] carries no information
beyond the mean, and clamp(Linear([mean, mean^2])) is a fixed quadratic in the
single mean-pool scalar — it collapses to a mean-only readout (no diffuseness
DOF). a72 vs a73 therefore isolates EXACTLY "does relational density-CONTRAST
(the spread) add over the mean?", with everything else (the warm-started axis,
the linear projection, the 2-in head) held fixed.

Why this is NOT a refuted family
--------------------------------
- NOT a59/a62 consensus (1st-order): those contrast each patch to the bag MEAN
  (a centroid reference) and re-weight a feature mean. a72 measures the FULL
  pairwise spread of the 1-D scores — no centroid, no re-weighting, no feature
  mean; D is symmetric in ALL pairs.
- NOT coverage / extent / threshold / fraction (a52/a54/a55): nothing counts a
  fraction above tau; D is a continuous spread of the value distribution.
- NOT mean-pool / a53 / a57 (LINEAR): D = mean-abs-pairwise-difference is
  2nd-order in s and not reconstructable from mean_s.
- NOT a free / gated attention scorer (baseline, a25, DE22): no per-patch weight
  at all; every (i,j) pair contributes symmetrically.
- NOT ||h||-norm weighting (DE15, a45): s_i is a signed inner product with the
  fibrosis direction; ||f_i|| is never computed or used.
- NOT a per-patch NONLINEAR severity MLP pooled to its mean (a56-a63): per-patch
  map is a single LINEAR projection; the nonlinearity lives only at the BAG
  level (pairwise |.|).
- NOT a sigma/softmax GATE between bag rep and y (DE11-13): the readout is a
  single unbounded Linear on [mean, D]; clamp only at the very end.

Seed-robustness argument
------------------------
1. Warm-start anchor: v starts at the prototype `axis` (+0.84 held-out coverage
   Spearman; monotone per-grade projections). The optimiser begins grade-aligned
   and only refines one 1280-d direction.
2. Minimal capacity: trainable = v (1280) + Linear(2->1) (3) = 1283, ~150x below
   the 197K that overfit this cohort. No bottleneck MLP to memorise bags.
3. Permutation-invariant (mean and the sorted-order Gini are symmetric over
   patches) and bag-size-invariant (both mean and D normalise by N / N^2;
   duplicating the bag leaves mean, D and y unchanged). Deterministic at
   inference. Degenerate N=1 -> D=0 (safe; reduces to the mean term).

Ablation companion: a73_fibrosis_pairwise_product.py — flips `kernel='product'`
(phi = s_i*s_j -> D = mean^2), collapsing the relational term to a pure function
of the mean. Removes EXACTLY the active ingredient (the density-contrast) with
the axis, projection, and head held fixed.

Kill criterion: abandon if a72 val_qwk < 0.78 at seed=2 AND a72 <= a73 (the
spread adds nothing over the mean). DoD = multi-seed audit {0,1,2,3,42}, not
seed=2 alone.

Param count (input_dim=1280): v 1280 + Linear(2,1) weight 2 + bias 1 = 1283.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> torch.Tensor:
    """Return the unit fibrosis axis from the train-patient prototype cache.

    Defensive: if the cache is missing, fall back to a deterministic random unit
    axis so the module still instantiates and the interface is sane (the
    warm-start benefit is lost, but the trainer never breaks). Built from seed=2
    TRAIN patients only (no val/test leakage).
    """
    if not path.is_file():
        g = torch.Generator().manual_seed(2)
        axis = torch.randn(input_dim, generator=g)
        return axis / axis.norm().clamp(min=1e-8)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    axis = blob["axis"].float()
    axis = axis / axis.norm().clamp(min=1e-8)
    assert axis.shape[0] == input_dim, f"axis dim {axis.shape[0]} != input_dim {input_dim}"
    return axis


def _mean_abs_pairwise_diff(s: torch.Tensor) -> torch.Tensor:
    """Mean absolute pairwise difference (Gini-style spread) of a 1-D bag.

    D = mean_{i,j} |s_i - s_j| over ALL N^2 ordered pairs (the i=j diagonal
    contributes 0). Computed in O(N log N) from a single sort using the exact
    identity for the mean absolute difference of a finite 1-D set:

        sum_{i,j} |s_i - s_j| = 2 * sum_k (2k - N + 1) * s_(k)

    with s_(k) the ascending order statistics (k = 0..N-1). Dividing by N^2
    gives the mean over ordered pairs. This is exact, permutation-invariant, and
    bag-size-invariant; it avoids materialising the N^2 difference matrix.
    """
    n = s.numel()
    if n <= 1:
        return s.new_zeros(())
    s_sorted, _ = torch.sort(s)
    k = torch.arange(n, device=s.device, dtype=s.dtype)
    coef = 2.0 * k - (n - 1)                     # (2k - N + 1), ascending order
    total = 2.0 * (coef * s_sorted).sum()        # = sum_{i,j} |s_i - s_j|
    return total / (n * n)                       # mean over all N^2 ordered pairs


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        kernel: str = "spread",         # "spread" (a72 main, |s_i-s_j|) | "product" (a73 abl., s_i*s_j)
        warm_start: bool = True,        # init v at the prototype fibrosis axis
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert kernel in ("spread", "product"), kernel
        self.kernel = kernel
        self.clamp_output = clamp_output

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        axis = _load_fibrosis_axis(path, input_dim)

        if warm_start:
            v0 = axis.clone()
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v0 = torch.randn(input_dim, generator=g)
            v0 = v0 / v0.norm().clamp(min=1e-8)

        # Learnable fibrosis direction (unit warm-start; raw input space).
        self.v = nn.Parameter(v0)

        # Tiny linear readout over [mean_s, D]. Init: weight=[1,0], bias=1.5 so
        # the model STARTS at a pure mean-pool readout (the spread term carries
        # zero weight) and must EARN the relational density-contrast — no
        # capacity confound versus the a73 product ablation.
        self.head = nn.Linear(2, num_classes)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.weight[0, 0] = 1.0   # weight on mean_s
            self.head.bias.fill_(1.5)      # interior of [0,3]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        # ONLY 'features' is consumed — no label, no val/test data (no leakage).
        s = features @ self.v                        # [N] per-patch density score (LINEAR)
        mean_s = s.mean()                            # 1st-order mean density

        if self.kernel == "spread":
            # 2nd-order relational density-CONTRAST: mean-abs-pairwise-difference
            # (Gini spread). NOT a function of the mean.
            D = _mean_abs_pairwise_diff(s)
        else:  # "product": phi = s_i * s_j -> D = mean^2 (pure function of the mean)
            D = mean_s * mean_s

        u = torch.stack([mean_s, D]).view(1, 2)      # [1, 2] bag descriptor
        y = self.head(u)                             # [1, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, s.detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    kernel="spread",         # a72 main = mean-abs-pairwise-difference (2nd-order spread)
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
