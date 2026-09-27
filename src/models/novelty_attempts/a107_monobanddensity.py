"""a107 — MonoBandDensity: monotone non-negative fibrosis-density-histogram readout.

ONE LINE
    A monotone, non-negative "fibrosis-density histogram" readout: project each
    patch onto a FROZEN train-only grade-axis, build a soft within-bag occupancy
    histogram over that 1-D axis (bag-size-invariant), and read the bag grade as
    a CUMULATIVE-MONOTONE non-negative weighting of the histogram bins — richer
    than mean-pool (it sees the whole density profile, not just its centroid)
    but the only free DOF are ~B sign- and order-constrained scalars that cannot
    rotate into a fold-specific nuisance direction.

WHY THIS DESIGN (vs the documented traps)
  * UNDERFIT trap (a101/a102/a106, 2–6 params): those collapse the bag to ONE
    scalar (a single projection mean / threshold) and cannot represent WHERE on
    the grade axis the tissue density sits. MonoBandDensity keeps the FULL B=16
    density profile p and reads a 16-term monotone functional of it, so it is
    strictly richer than a single mean/quantile — it can express "G2 vs G3 = how
    much mass sits in the top intensity bins", exactly the distinction the
    underfitters miss. 18 DOF sits well above the 2–6 underfit floor.
  * VAL-OVERFIT trap (a91/a93/a99, 164K params): those lift seed=2 val then tank
    test because their FREE 1280->128 bottleneck keys on fold-specific nuisance
    directions — the variance lives in that free linear map, not the pool it
    decorates. MonoBandDensity DELETES that free map: the only thing that touches
    the 1280-d features is a FROZEN unit axis (zero DOF, cannot rotate), and the
    only learnable objects are 18 scalars that are sign-constrained (non-negative
    via softplus) and order-constrained (cumulative => monotone non-decreasing).
    With 18 monotone-locked DOF the model literally cannot represent a
    fold-specific anti-grade reweighting of the density profile, so it cannot
    manufacture the seed=2 val lift that richer mechanisms buy by overfitting.

MECHANISM (single bag f in R^{N x 1280}, RAW scalar out)
  CONSTANTS (frozen buffers, train-only seed=2, NO grad):
    a   = unit grade axis = (c3 - c0)/||.|| from prototypes['axis'] in R^1280.
    m0,v0 = mean/std of the 4 train-only class-prototype projections <c_k, a>
            (a fold-invariant, train-only calibration; falls back to this-bag
            stats only if the cache is absent).
    mu_b (b=1..B, B=16) = fixed bin centres on the CALIBRATED axis, a linspace
            spanning the (calibrated) prototype range with a small margin.
    sigma = median spacing of mu (a frozen scalar RBF bandwidth).
  STEP 1 — frozen 1-D grade coordinate (NO free DOF):
    s_i = <f_i, a> / ||a||                 # [N] scalar per patch
    s_i <- (s_i - m0) / v0                  # FROZEN calibration => unit-ish scale
    No learnable rotation: the scoring direction is locked to the grade axis.
  STEP 2 — soft occupancy histogram (parameter-free, perm- & size-invariant):
    k_{i,b} = exp(-(s_i - mu_b)^2 / (2 sigma^2))                       # [N,B] RBF
    p_b = (1/N) * sum_i k_{i,b} / (sum_{b'} k_{i,b'} + eps)            # [B] density
    => p is a normalised within-bag density over the grade axis: "what fraction
    of marrow tissue sits at each fibrosis-intensity level?". Mean over patches
    => bag-size invariant; sum over i => permutation-invariant.
  STEP 3 — HARD-CONSTRAINED monotone readout (the ONLY learnable params):
    P_b = sum_{b'<=b} p_{b'}                # CDF over the grade axis in [0,1]
    w_b = softplus(theta_b)                 # B non-negative weights (free)
    g_b = cumsum(w_b) ascending             # => g_1<=...<=g_B  MONOTONE NON-DEC
    g_b <- g_b / (g_B + eps)                # weight-tied: top bin weight = 1
    area = (1/B) * sum_b g_b * (1 - P_{b-1})  # B-invariant exceedance area in ~[0,1]
    y = bias + scale * area                 # scale = softplus(rho) > 0
    i.e. y is a non-negative, monotone-increasing functional of the fraction of
    tissue ABOVE each intensity level. Because g is forced non-decreasing and
    non-negative and scale>0, y is provably MONOTONE in "mass shifted to higher
    bins": it cannot learn a non-monotone (anti-grade) response, and cannot
    up-weight a low-intensity bin to chase a fold-specific quirk.

PARAMS (only these learn): theta in R^B (B=16), rho (scale), bias => 18 scalars.
  NOTHING ELSE. No 1280->128 bottleneck, no attention, no learnable projection.
  The 163,968-param free linear map every tied mechanism carries — where the
  variance actually lives — is REMOVED and replaced by a frozen axis + 18
  shape-constrained scalars.

Warm-start: theta init so w_b ~ uniform (g linear) => y starts as "mean grade
coordinate", a sane diffuse prior; rho,bias init to span ~[0,3].

RAW logits out (no clamp). Deterministic at eval (no dropout, all-frozen
buffers). Permutation- & bag-size-invariant by construction.

LEAKAGE: the warm axis (and the prototype-derived calibration) is used ONLY as
a FROZEN buffer that is the SAME physical object on every fold — it injects no
per-fold information (unlike a head-init only the seed=2 optimiser exploits, see
a103). forward() reads ONLY 'features'.

Ablation a108: flip monotone=False so the 16 readout weights become FREE
(sign- and order-unconstrained, g_b = theta_b) and scale is free-sign too,
holding the frozen axis, B=16 histogram, normalisation and param count
byte-identical. Isolates: "does the hard monotone/non-negative shape constraint
prevent the val-overfit a free 16-weight linear readout of the same density
would suffer?".

Param count (B=16): theta(16) + rho(1) + bias(1) = 18.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"

_EPS = 1e-8


def _inv_softplus(x: float) -> float:
    """Inverse softplus: r s.t. softplus(r) == x (x > 0)."""
    import math

    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def _load_axis_and_calib(
    path: Path, input_dim: int
) -> Tuple[Optional[torch.Tensor], Optional[float], Optional[float], Optional[Tuple[float, float]]]:
    """Return (unit_axis, m0, v0, (lo, hi)) from train-only seed=2 prototypes.

    All quantities are derived ONLY from the seed=2 train file:
      - axis: the stored unit grade axis = (c3 - c0)/||.||
      - m0,v0: mean/std of the 4 class-prototype projections <c_k, axis>
               (a fold-invariant train-only calibration of the projection scale)
      - (lo, hi): min/max prototype projection in CALIBRATED coords (bin span)
    Returns (None, None, None, None) if the cache is absent so the model still
    constructs and falls back to deterministic per-bag stats. No test info, no
    runtime features.
    """
    if not path.is_file():
        return None, None, None, None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None, None, None
    axis = blob.get("axis", None)
    if axis is None:
        return None, None, None, None
    axis = axis.float().view(-1)
    if axis.numel() != input_dim:
        return None, None, None, None
    axis = axis / axis.norm().clamp(min=_EPS)

    protos = blob.get("prototypes", None)
    if not isinstance(protos, dict) or len(protos) == 0:
        return axis, None, None, None
    # raw projections of the train-only class centroids onto the unit axis
    projs = []
    for k in sorted(protos.keys()):
        c = protos[k].float().view(-1)
        if c.numel() != input_dim:
            return axis, None, None, None
        projs.append(float((c * axis).sum()))
    projs_t = torch.tensor(projs, dtype=torch.float32)
    m0 = float(projs_t.mean())
    v0 = float(projs_t.std(unbiased=False).clamp(min=_EPS))
    # calibrated prototype coords -> span for bin grid
    calib = (projs_t - m0) / v0
    lo, hi = float(calib.min()), float(calib.max())
    return axis, m0, v0, (lo, hi)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        n_bins: int = 16,            # B: number of soft-histogram bins
        monotone: bool = True,       # a108 ablation flips this to False (free weights)
        bin_margin: float = 0.5,     # extra margin (in calibrated units) on bin span
        prototype_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert n_bins >= 1, "n_bins (B) must be >= 1."
        self.input_dim = int(input_dim)
        self.n_bins = int(n_bins)
        self.monotone = bool(monotone)

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        axis, m0, v0, span = _load_axis_and_calib(path, input_dim)

        # --- FROZEN grade axis (zero DOF, cannot rotate) ---
        if axis is None:
            # Fall back to a fixed (deterministic, seeded) random unit vector so
            # the module still constructs without the cache. Real runs use the
            # cached train-only axis.
            g = torch.Generator().manual_seed(2)
            axis = torch.randn(input_dim, generator=g)
            axis = axis / axis.norm().clamp(min=_EPS)
        self.register_buffer("axis", axis)  # [D] frozen unit grade axis

        # --- FROZEN calibration (train-only prototype-projection mean/std) ---
        # If absent, m0/v0 are NaN sentinels => forward calibrates per-bag
        # deterministically (mean/std of THIS bag's projections).
        self.register_buffer(
            "m0", torch.tensor(float("nan") if m0 is None else m0, dtype=torch.float32)
        )
        self.register_buffer(
            "v0", torch.tensor(float("nan") if v0 is None else v0, dtype=torch.float32)
        )
        self._have_calib = m0 is not None and v0 is not None

        # --- FROZEN bin centres on the calibrated axis ---
        if span is not None:
            lo, hi = span
            lo, hi = lo - bin_margin, hi + bin_margin
        else:
            # No prototype span available: a fixed symmetric grid in calibrated
            # units. forward's per-bag calibration makes this still meaningful.
            lo, hi = -2.5, 2.5
        if self.n_bins == 1:
            mu = torch.tensor([(lo + hi) / 2.0], dtype=torch.float32)
            sigma = float(max(hi - lo, _EPS))
        else:
            mu = torch.linspace(lo, hi, self.n_bins, dtype=torch.float32)
            spacing = (mu[1:] - mu[:-1]).abs()
            sigma = float(spacing.median().clamp(min=_EPS))
        self.register_buffer("mu", mu)                                  # [B]
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float32))

        # --- LEARNABLE params (the ONLY free DOF) ---
        # theta -> w (per-bin weight). Warm-start so the readout starts ~ "mean
        # grade coordinate": monotone => g linear in b (uniform w via softplus);
        # free => g_b ~ b/B linear ramp (same starting functional).
        if self.monotone:
            # softplus(theta) ~ const => g = cumsum(const) is a linear ramp.
            # softplus(0) = ln2 ~ 0.693 (a fine uniform start).
            self.theta = nn.Parameter(torch.zeros(self.n_bins))
            # scale = softplus(rho) > 0 ; rho s.t. softplus(rho) ~ 3 to span [0,3].
            self.rho = nn.Parameter(torch.tensor(float(_inv_softplus(3.0))))
        else:
            # Free, sign- and order-UNCONSTRAINED weights. Init to the SAME
            # starting functional g_b = b/B linear ramp so a107/a108 start
            # identically (only the constraint, not the init, differs).
            init_g = torch.arange(1, self.n_bins + 1, dtype=torch.float32) / float(self.n_bins)
            self.theta = nn.Parameter(init_g)
            # free-sign scale init to +3 (matches monotone's softplus(rho)~3).
            self.rho = nn.Parameter(torch.tensor(3.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        f = features
        if f.dim() == 1:
            f = f.unsqueeze(0)

        # STEP 1 — frozen 1-D grade coordinate (no free DOF). axis is unit-norm
        # already; divide by its norm to match the spec exactly (== identity).
        s = (f * self.axis).sum(dim=1) / self.axis.norm().clamp(min=_EPS)  # [N]

        if self._have_calib and torch.isfinite(self.m0) and torch.isfinite(self.v0):
            m0, v0 = self.m0, self.v0
        else:
            # Deterministic per-bag fallback: mean/std of THIS bag's projections.
            m0 = s.mean().detach()
            v0 = s.std(unbiased=False).clamp(min=_EPS).detach()
        s = (s - m0) / v0                                                  # [N]

        # STEP 2 — soft occupancy histogram (parameter-free; perm/size-invariant)
        # k_{i,b} = exp(-(s_i - mu_b)^2 / (2 sigma^2))
        d2 = (s.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2                  # [N, B]
        k = torch.exp(-d2 / (2.0 * self.sigma * self.sigma))              # [N, B]
        # per-patch normalise across bins -> soft assignment, then mean over i
        k = k / (k.sum(dim=1, keepdim=True) + _EPS)                       # [N, B]
        p = k.mean(dim=0)                                                 # [B] density

        # STEP 3 — readout
        P = torch.cumsum(p, dim=0)                                        # [B] CDF
        # exceedance: fraction of mass ABOVE each bin = 1 - P_{b-1}
        # P_{b-1}: shift right, P_0 := 0
        P_prev = torch.cat([torch.zeros(1, device=P.device, dtype=P.dtype), P[:-1]], dim=0)
        exceed = 1.0 - P_prev                                             # [B]

        if self.monotone:
            w = F.softplus(self.theta)                                    # [B] >= 0
            g = torch.cumsum(w, dim=0)                                    # [B] non-decreasing
            g = g / (g[-1] + _EPS)                                        # tie: top weight = 1
            scale = F.softplus(self.rho)                                  # > 0
        else:
            # FREE, sign- and order-UNCONSTRAINED 16-weight linear readout.
            g = self.theta                                                # [B] free
            scale = self.rho                                              # free-sign

        # Normalise the exceedance area by B so the readout magnitude is
        # bin-count-invariant (the functional's natural scale would otherwise
        # grow ~linearly in B). A frozen constant divisor: it changes NO DOF,
        # NO monotonicity, NO invariance — only keeps the warm-start in-range.
        area = (g * exceed).sum() / float(self.n_bins)                    # scalar
        y = self.bias + scale * area                                      # scalar RAW

        logits = y.view(1)
        if return_attention:
            return logits, None, None
        return logits, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    n_bins=16,             # B = 16 soft-histogram bins
    monotone=True,         # a107 main = hard monotone non-negative readout
    bin_margin=0.5,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
)
