"""a70 — SOFT-QUANTILE VECTOR of the fibrosis projection (grading-aligned,
multi-quantile readout of ONE warm-started axis).

LENS
----
Reticulin grade = OVERALL / DIFFUSE reticulin density across the whole bag.
Project every patch onto the warm-started fibrosis axis to get a per-patch
score s_i = <f_i, v> (a signed 1-D position on the G0..G3 ordinal direction),
then read a SMALL SET OF BAG QUANTILES of the value-distribution {s_i} at FIXED
levels {0.1, 0.5, 0.9}. The whole nonlinearity lives in the BAG-LEVEL quantile
statistic; the per-patch map is a single LINEAR projection.

Hypothesis (Hn_fibrosis_quantiles)
----------------------------------
Different grade boundaries are read off DIFFERENT parts of the same density
distribution:
  * G0|G1 is decided by the UPPER TAIL rising off the clean-marrow floor
    (q0.9 lifts even when most tiles are still clean);
  * G2|G3 is decided by the MEDIAN / upper-mid of the distribution rising
    (the meshwork becomes diffuse, so the bulk — not just the tail — moves up).
Reading multiple quantiles of ONE learned axis behaves like having a separate
per-boundary axis WITHOUT learning three directions: a single 1280-d direction
plus a 3->1 linear combination of its {0.1, 0.5, 0.9} quantiles.

Mechanism (exact forward-pass math; single bag, features f in R^{N x 1280})
---------------------------------------------------------------------------
  v                                       # warm-started fibrosis axis (learnable Parameter, raw 1280-d, NO bottleneck)
  s_i = <f_i, v>                          # [N] per-patch fibrosis projection (1-D, LINEAR)
  # Differentiable quantile via a softmax-weighted order statistic. Build a
  # soft (differentiable) rank in [0,1] for every patch, then for each level L
  # weight patches whose soft-rank is near L and read the weighted score:
  r_i = (#{j : s_j < s_i}) / (N - 1)      # soft rank in [0,1] (differentiable; see below)
  w^L_i = softmax_i( -(r_i - L)^2 / tau ) # peak near the patches at fractional rank L
  q_L = sum_i w^L_i * s_i                  # soft L-quantile of {s_i}
  u   = [q_0.1, q_0.5, q_0.9]              # 3-vector of quantiles of ONE axis
  y   = clamp( Linear(3 -> 1)(u) , 0, 3 )  # tiny linear readout

Soft rank. r_i is built differentiably from pairwise comparisons softened by a
temperature: r_i = mean_j sigmoid((s_i - s_j) / rank_tau), scaled to span [0,1].
This is a standard differentiable-ranking surrogate; as rank_tau -> 0 it
recovers the hard fractional rank. It is permutation-symmetric (r_i depends only
on the multiset {s_j}) so the resulting quantiles are permutation-invariant. It
is also (approximately) bag-size-invariant: r_i and the softmax weights are
normalised over the bag, so duplicating the bag leaves {q_L} and y unchanged in
the rank_tau -> 0 / tau -> 0 limit and very nearly so for finite temperatures.

Why the recurring "collapses to mean-pool linear ablation" failure is
STRUCTURALLY IMPOSSIBLE here
----------------------------------------------------------------------
The per-patch map is a LINEAR projection s_i = <f_i, v>. A pure mean-pool +
linear head can only produce a function of mean_i s_i = <mean_i f_i, v>. The
soft quantiles q_0.1 and q_0.9 are NOT linear functionals of the mean: they
depend on the ORDER STATISTICS of {s_i} (which patch sits in the tail), an
operation mean o linear cannot absorb. So a70 reading three quantiles is not
expressible as Linear(mean_i f_i). The ablation that keeps ONLY q_0.5 (the
soft median == a robust central density) removes EXACTLY the tail quantiles —
isolating "do the tail quantiles add over the central density?" — and starts
the head at the same readout (no capacity confound).

Why this is NOT a refuted family
--------------------------------
- NOT coverage / extent / threshold / fraction-of-positives (a52/a54/a55):
  nothing counts a fraction above tau; we read VALUES at fixed rank positions.
- NOT mean-pool / a53 / a57 (LINEAR): q_0.1, q_0.9 are order statistics, not
  linear functionals of the mean projection.
- NOT moments (a66): we read fixed-LEVEL quantiles (rank-based), not central
  moments (value-based variance/skew).
- NOT a free / gated attention scorer (baseline, a25, DE22): the per-patch
  weights are FIXED functions of the rank (softmax of -(r_i - L)^2/tau), not a
  learned content scorer; no head concentrates on a few patches by learned gate.
- NOT ||h||-norm weighting (DE15, a45): s_i is a signed inner product with the
  fibrosis direction; ||f_i|| is never computed or used.
- NOT a per-patch NONLINEAR severity MLP pooled to its mean (a56-a63): per-patch
  map is a single LINEAR projection; the nonlinearity (ranking + softmax) lives
  only in the BAG-LEVEL quantile computation.

Seed-robustness argument
-------------------------
1. Warm-start anchor: v starts at the prototype `axis` (+0.84 held-out coverage
   Spearman), so the optimiser begins grade-aligned and only refines ONE 1280-d
   direction.
2. Minimal capacity: trainable = v (1280) + Linear(3->1) (4) = 1284, ~150x below
   the 197K that overfit this cohort. No bottleneck MLP to memorise bags. tau and
   rank_tau are FIXED hyper-parameters (registered as buffers, not learned).
3. Permutation-invariant (soft rank + softmax-weighted sum are symmetric over
   patches) and ~bag-size-invariant (rank and weights normalise over the bag).
   Deterministic at inference. Degenerate N=1 -> r=0.5, every q_L = s_0 (safe).

Ablation companion: a71_fibrosis_quantiles_median_only.py — flips
`quantile_levels=[0.5]` so the readout is Linear(1->1)(q_0.5) (the soft median,
a robust central density of the projection). Removes EXACTLY the active
ingredient (the upper/lower tail quantiles) with everything else held fixed.
a70 vs a71 isolates "do the tail quantiles add over the central density?".

Kill criterion: abandon if a70 val_qwk < 0.78 at seed=2 AND a70 <= a71. DoD =
multi-seed audit {0,1,2,3,42}, not seed=2 alone.

Param count (input_dim=1280, 3 quantiles): v 1280 + Linear(3,1) weight 3 +
bias 1 = 1284. Ablation a71 (1 quantile): v 1280 + Linear(1,1) 1 + 1 = 1282.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> torch.Tensor:
    """Return the unit warm-start fibrosis axis from the train-patient cache.

    Defensive: if the cache is missing, fall back to a deterministic random unit
    axis so the module still instantiates and the trainer never breaks (the
    warm-start benefit is lost, but the interface stays sane).
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


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        quantile_levels: Sequence[float] = (0.1, 0.5, 0.9),  # a70 main; a71 ablation = (0.5,)
        tau: float = 0.05,              # softmax temperature over (soft_rank - L)^2 (FIXED)
        rank_tau: float = 0.05,         # temperature for the differentiable soft rank (FIXED)
        warm_start: bool = True,        # init v at the prototype fibrosis axis
        prototype_path: Optional[str] = None,
        random_seed: int = 2,           # used only if warm_start=False
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        levels = [float(L) for L in quantile_levels]
        assert len(levels) >= 1, "need at least one quantile level"
        assert all(0.0 <= L <= 1.0 for L in levels), f"levels must be in [0,1]: {levels}"
        self.clamp_output = clamp_output

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        axis = _load_fibrosis_axis(path, input_dim)
        if warm_start:
            v0 = axis.clone()
        else:
            g = torch.Generator().manual_seed(int(random_seed))
            v0 = torch.randn(input_dim, generator=g)
            v0 = v0 / v0.norm().clamp(min=1e-8)

        # Learnable fibrosis direction (unit warm-start; raw input space, NO bottleneck).
        self.v = nn.Parameter(v0)

        # Fixed quantile levels + temperatures (buffers: move with device, no grad,
        # never per-seed degrees of freedom).
        self.register_buffer("levels", torch.tensor(levels))  # [Q]
        self.register_buffer("tau", torch.tensor(float(tau)))
        self.register_buffer("rank_tau", torch.tensor(float(rank_tau)))

        # Tiny linear readout over the quantile vector. Init: equal weight on each
        # quantile (so q's are averaged ~ robust mean) and an interior bias, so the
        # model starts near a sensible central-density readout.
        n_q = len(levels)
        self.head = nn.Linear(n_q, num_classes)
        with torch.no_grad():
            self.head.weight.fill_(1.0 / float(n_q))
            self.head.bias.fill_(1.5)  # interior of [0,3]

    @property
    def quantile_levels(self) -> List[float]:
        return [float(x) for x in self.levels.tolist()]

    def _soft_rank(self, s: torch.Tensor) -> torch.Tensor:
        """Differentiable fractional rank r_i in [0,1] of each score s_i.

        r_i = mean_j sigmoid((s_i - s_j) / rank_tau), then rescaled so it spans
        [0,1] (the diagonal sigmoid(0)=0.5 self-term is subtracted out). As
        rank_tau -> 0 this recovers the hard fractional rank #{j: s_j<s_i}/(N-1).
        Permutation-symmetric (depends only on the multiset {s_j}).
        """
        n = s.shape[0]
        if n == 1:
            return s.new_full((1,), 0.5)
        diff = s.unsqueeze(1) - s.unsqueeze(0)            # [N, N], diff[i,j] = s_i - s_j
        comp = torch.sigmoid(diff / self.rank_tau)        # [N, N]
        # Sum over j excluding the self-term (sigmoid(0)=0.5 on the diagonal).
        raw = comp.sum(dim=1) - 0.5                        # [N], in [0, N-1]
        r = raw / float(n - 1)                             # [N] in [0,1]
        return r

    def _soft_quantiles(self, s: torch.Tensor) -> torch.Tensor:
        """Soft quantiles q_L = sum_i softmax_i(-(r_i - L)^2 / tau) * s_i for each L."""
        r = self._soft_rank(s)                             # [N]
        # [Q, N]: squared distance of each patch rank to each target level.
        d2 = (r.unsqueeze(0) - self.levels.unsqueeze(1)) ** 2   # [Q, N]
        w = F.softmax(-d2 / self.tau, dim=1)                    # [Q, N] weights per level
        q = w @ s                                               # [Q] soft quantiles
        return q

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract). Uses ONLY
        # 'features' + the warm-started direction — no labels, no val/test data.
        s = features @ self.v                # [N] per-patch fibrosis projection (LINEAR)
        q = self._soft_quantiles(s)          # [Q] soft quantile vector of ONE axis
        y = self.head(q.view(1, -1))         # [1, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, s.detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    quantile_levels=(0.1, 0.5, 0.9),   # a70 main = tail + median quantiles of the projection
    tau=0.05,
    rank_tau=0.05,
    warm_start=True,
    prototype_path=None,               # None -> data/prototypes_virchow2_reti_train_seed2.pt
    random_seed=2,
    clamp_output=True,
)
