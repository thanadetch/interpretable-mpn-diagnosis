"""a74 — Fibrosis DISTRIBUTION-MATCHING MIL (1-D Wasserstein / quantile-match
to per-grade reference distributions).

Hypothesis (Hn_fibrosis_dist_match)
-----------------------------------
Reticulin grade = the OVERALL / DIFFUSE density distribution of the fibre
meshwork along the prototype fibrosis axis — NOT the mean density, NOT a few
hot patches, but the *shape of the whole per-patch score distribution*. Two
bags can share the same MEAN projection yet differ in grade because their
score distributions differ (a focal hot tail vs a uniformly-moderate field).
So grade a bag by WHICH per-grade reference DISTRIBUTION its own score
distribution most resembles — a 1-D quantile (Wasserstein-style) match — and
read out the soft-argmin expected grade.

Mechanism (single bag, features f in R^{N x 1280})
---------------------------------------------------
  v_hat            : warm-started fibrosis axis (unit, learnable Parameter)
  s_i = <f_i, v>   : [N] per-patch fibrosis projection (1-D, LINEAR — frozen-form map)
  Q_bag[q]         : [Q] the bag's OWN soft quantiles of {s_i} at fixed levels
                     p_1..p_Q (differentiable soft-rank quantile, sort-free,
                     permutation- & bag-size-invariant; degenerate-N safe).
  R_g[q]           : [4, Q] per-grade REFERENCE quantile vectors, precomputed
                     OFFLINE from seed=2 TRAIN patches ONLY (see LEAKAGE rule).
  d_g = mean_q (Q_bag[q] - R_g[q])^2          # squared quantile distance to grade g
                     (a discretised 1-D Wasserstein-2 / quantile-matching cost)
  y   = clamp( sum_g g * softmax(-d_g / tau) , 0, 3 )   # soft-argmin expected grade

`tau` (a single positive learnable temperature) is the ONLY non-axis trainable
scalar beyond v_hat — capacity is minimal. The readout is the expected grade
under the soft assignment that puts mass on the grade whose reference
DISTRIBUTION the bag matches.

Why the per-patch map is a FROZEN-FORM LINEAR projection and the nonlinearity
lives ONLY in the BAG-LEVEL statistic
-----------------------------------------------------------------------------
s_i = <f_i, v> is a single linear projection. The quantiles Q_bag[q] are a
NONLINEAR bag-level functional of the distribution of {s_i} (order statistics /
soft ranks), and the distance d_g + softmax readout is nonlinear in those
quantiles. Mean-pool (a53/a57) reduces the bag to <mean_i f_i, v> = mean_i s_i,
i.e. ONE number (the mean of the distribution). A74 reads Q quantiles + matches
to full reference distributions, so it CANNOT collapse to mean-pool the way the
a56-a63 per-patch-NONLINEAR severity pools did (where the ablation beat the main
6 times): there is no per-patch nonlinearity to absorb into the mean. The
ablation a75 collapses each reference R_g to its MEAN and matches only the bag
mean — that is the mean-pool 1-D nearest-prototype-mean classifier, the EXACT
"mean-only" baseline this design must beat.

LEAKAGE (critical)
------------------
R_g are read from data/fibrosis_refs_seed2.pt, built OFFLINE by
scripts/build_fibrosis_refs.py from the seed=2 TRAIN split ONLY (it calls the
locked patient_split(full_dataset, seed=2) and uses ONLY train_idx bags; val/
test patches are NEVER touched). forward() consumes ONLY `features` plus these
precomputed train-only references — no labels, no val/test data. Loaded
DEFENSIVELY: if the cache is missing/malformed, fall back to a deterministic
synthetic monotone reference (warm-start benefit lost, interface stays sane).
The reference is seed=2-specific by construction (file carries 'seed':2).

Why this is NOT a refuted family
--------------------------------
- NOT coverage / threshold / fraction (a52/a66): nothing counts a fraction
  above tau; we read the bag's quantile vector and match distributions.
- NOT mean-pool / a53 / a57: the match uses Q quantiles, not just the mean
  (a75 ablation isolates exactly the full-distribution-vs-mean question).
- NOT moments (a64/a66): we compare the bag's quantiles to fixed per-grade
  reference quantiles (a distribution distance), not low-order central moments.
- NOT a free attention scorer / gated attention: no learned per-patch weight;
  the soft-rank weights are a fixed differentiable quantile estimator.
- NOT ||h||-norm weighting (a45): s_i is a signed inner product with the
  fibrosis direction; ||f_i|| is never computed or used.
- NOT a per-patch NONLINEAR severity MLP pooled to its mean (a56-a63): the
  per-patch map is a single LINEAR projection.

Permutation-invariant (quantiles are symmetric in patch order) and
bag-size-invariant (soft ranks normalise by N; quantiles depend only on the
distribution, not N). Deterministic at inference. Degenerate N=1 -> all
quantiles equal s_0 (safe; reduces to nearest-prototype on the single score).

Ablation companion: a75_fibrosis_dist_match_meanonly.py — flips
`match_mode='mean'`, collapsing each reference to its MEAN R_g_mean and the bag
to its mean score, so d_g = (mean_s - R_g_mean)^2. Removes EXACTLY the active
ingredient (matching the full DISTRIBUTION) -> a nearest-prototype-MEAN 1-D
classifier on the bag mean.

Kill criterion: abandon if a74 val_qwk < 0.78 at seed=2 AND a74 <= a75 (the
full distribution match adds nothing over the mean match). DoD = multi-seed
audit {0,1,2,3,42}, not seed=2 alone.

Param count (input_dim=1280): v_hat 1280 + log_tau 1 = 1281.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"
_DEFAULT_REFS_PATH = _REPO_ROOT / "data" / "fibrosis_refs_seed2.pt"

# Fixed quantile levels p_1..p_Q (interior, avoids the 0/1 endpoints that are
# unstable for small N). MUST match the levels used in build_fibrosis_refs.py.
_DEFAULT_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _load_fibrosis_axis(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis from the train-patient prototype cache (defensive).

    Returns None (caller falls back to a deterministic random axis) if the cache
    is missing or malformed, so the module never hard-crashes a screen.
    """
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        axis = blob["axis"].float()
        axis = axis / axis.norm().clamp(min=1e-8)
        if axis.shape[0] != input_dim:
            return None
        return axis
    except Exception:
        return None


def _load_references(
    path: Path, levels: Tuple[float, ...]
) -> Optional[torch.Tensor]:
    """Per-grade reference quantile matrix R in R^{4 x Q} (TRAIN-only, seed=2).

    Loaded DEFENSIVELY from data/fibrosis_refs_seed2.pt (built offline by
    scripts/build_fibrosis_refs.py). Returns None if missing/malformed/level-
    mismatched so the caller can fall back to a synthetic monotone reference.

    The file's 'R' maps grade g -> [Q] reference quantiles, and its 'levels'
    must match `levels` exactly (so the bag's own quantiles are comparable).
    """
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        file_levels = tuple(float(x) for x in blob["levels"])
        if file_levels != tuple(float(x) for x in levels):
            return None
        R = torch.stack([blob["R"][g].float() for g in range(4)], dim=0)  # [4, Q]
        if R.shape != (4, len(levels)):
            return None
        return R
    except Exception:
        return None


def _synthetic_references(levels: Tuple[float, ...]) -> torch.Tensor:
    """Deterministic monotone fallback references when the cache is missing.

    Grade means ramp 0->3 along the axis with a fixed spread, so distribution
    matching still does something sensible (the warm-start benefit is lost).
    """
    q = len(levels)
    lev = torch.tensor(levels, dtype=torch.float32)
    # Standard-normal-ish quantile offsets centred at 0 (probit-free approx).
    offsets = (lev - 0.5) * 4.0  # spread the levels around the grade mean
    grade_means = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    R = grade_means.view(4, 1) + offsets.view(1, q)  # [4, Q]
    return R


def _soft_quantiles(
    s: torch.Tensor, levels: torch.Tensor, rank_sigma: float, sel_tau: float
) -> torch.Tensor:
    """Differentiable, sort-free, permutation- & size-invariant soft quantiles.

    For each patch i, soft_rank_i = mean_j sigmoid((s_i - s_j)/rank_sigma) in
    [0,1] (its fractional position in the distribution). For target level p_q,
    the soft quantile is the score-weighted average where weights concentrate on
    patches whose soft rank is near p_q:
        w[q,i] = softmax_i( -(soft_rank_i - p_q)^2 / sel_tau )
        Q_bag[q] = sum_i w[q,i] * s_i.
    Depends only on the distribution of {s_i}; N=1 -> every quantile == s_0.
    """
    n = s.shape[0]
    if n == 1:
        return s.expand(levels.shape[0]).clone()
    # Soft (fractional) ranks in [0,1], symmetric over patch order.
    diff = (s.unsqueeze(1) - s.unsqueeze(0)) / rank_sigma   # [N, N]
    soft_rank = torch.sigmoid(diff).mean(dim=1)             # [N]
    # Selection weights: [Q, N] concentrating on rank ~ p_q.
    gap = soft_rank.unsqueeze(0) - levels.unsqueeze(1)      # [Q, N]
    logits = -(gap * gap) / sel_tau                         # [Q, N]
    w = torch.softmax(logits, dim=1)                        # [Q, N]
    return w @ s                                            # [Q]


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        match_mode: str = "dist",       # "dist" (a74 main) | "mean" (a75 ablation)
        levels: Tuple[float, ...] = _DEFAULT_LEVELS,
        warm_start: bool = True,        # init v_hat at the prototype fibrosis axis
        prototype_path: Optional[str] = None,
        refs_path: Optional[str] = None,
        rank_sigma: float = 1.0,        # soft-rank temperature (fixed, distribution scale)
        sel_tau: float = 0.02,          # quantile-selection sharpness (fixed)
        tau_init: float = 1.0,          # initial soft-argmin temperature (learnable, >0)
        random_seed: int = 2,           # used only if warm_start=False or cache missing
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert match_mode in ("dist", "mean"), match_mode
        self.match_mode = match_mode
        self.input_dim = int(input_dim)
        self.rank_sigma = float(rank_sigma)
        self.sel_tau = float(sel_tau)
        self.clamp_output = bool(clamp_output)

        levels_t = torch.tensor([float(x) for x in levels], dtype=torch.float32)
        self.register_buffer("levels", levels_t)  # [Q] fixed quantile levels
        self.n_levels = int(levels_t.numel())

        # --- learnable fibrosis direction (unit warm-start; raw input space) --
        axis = (
            _load_fibrosis_axis(
                Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH,
                self.input_dim,
            )
            if warm_start
            else None
        )
        if axis is None:  # deterministic random fall-back
            g = torch.Generator().manual_seed(int(random_seed))
            axis = torch.randn(self.input_dim, generator=g)
            axis = axis / axis.norm().clamp(min=1e-8)
        self.v = nn.Parameter(axis)

        # --- per-grade reference quantiles R in R^{4 x Q} (TRAIN-only, frozen) -
        # Loaded defensively from the OFFLINE seed=2-train cache. Registered as a
        # buffer (no gradient; moves with the module device). val/test patches
        # are NEVER used to build R (see scripts/build_fibrosis_refs.py).
        R = _load_references(
            Path(refs_path) if refs_path else _DEFAULT_REFS_PATH,
            tuple(float(x) for x in levels),
        )
        if R is None:
            R = _synthetic_references(tuple(float(x) for x in levels))
        self.register_buffer("R", R)               # [4, Q] full-distribution refs
        self.register_buffer("R_mean", R.mean(dim=1))  # [4] mean-collapsed refs (a75)

        # --- soft-argmin temperature (single positive learnable scalar) -------
        # parameterised in log-space so tau = exp(log_tau) stays > 0.
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau_init)))))

        # Grade values 0..3 for the expected-grade readout (fixed).
        self.register_buffer("grades", torch.arange(4, dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        # ONLY `features` is consumed (+ precomputed TRAIN-only references R).
        s = features @ self.v                          # [N] per-patch fibrosis projection
        tau = self.log_tau.exp().clamp(min=1e-3)       # > 0 soft-argmin temperature

        if self.match_mode == "dist":
            # Bag's OWN soft quantiles, matched to per-grade reference quantiles.
            q_bag = _soft_quantiles(s, self.levels, self.rank_sigma, self.sel_tau)  # [Q]
            d = ((q_bag.unsqueeze(0) - self.R) ** 2).mean(dim=1)   # [4] quantile dist
        else:  # "mean" ablation: match ONLY the bag mean to each grade's ref mean
            mean_s = s.mean()                                      # scalar bag mean
            d = (mean_s - self.R_mean) ** 2                        # [4] mean dist

        assign = torch.softmax(-d / tau, dim=0)        # [4] soft grade assignment
        y = (assign * self.grades).sum().view(1)       # soft-argmin expected grade [1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, assign.detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    match_mode="dist",       # a74 main = match the full quantile DISTRIBUTION
    levels=_DEFAULT_LEVELS,
    warm_start=True,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
    refs_path=None,          # None -> data/fibrosis_refs_seed2.pt (TRAIN-only, seed=2)
    rank_sigma=1.0,
    sel_tau=0.02,
    tau_init=1.0,
    random_seed=2,
    clamp_output=True,
)
