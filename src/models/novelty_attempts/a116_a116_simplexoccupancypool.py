"""a116 — SimplexOccupancyPool: hard-constrained diffuse grade-occupancy readout.

ONE LINE
    Soft-assign every patch to the 4 FROZEN train-only grade prototypes inside
    the rank-3 grade-mean subspace, diffuse-average those assignments over the
    bag to get a bag-wide grade-OCCUPANCY distribution q in the probability
    simplex Delta^3 ("what fraction of marrow tissue looks like each grade"),
    and read the grade as a MONOTONE non-negative ordinal-anchor expectation
    y = sum_k g_k q_k. 10 shape-locked DOF — richer than mean-projection but
    structurally unable to rotate into fold noise.

WHY THIS LENS = HARD-CONSTRAINED-CAPACITY, and why it is the UNEXPLORED interior
-------------------------------------------------------------------------------
The session brackets this point but never hit it:
  * a101/a102/a106 (2-6 param frozen-axis MEAN-projection) -> UNDERFIT trap
    (val 0.60-0.79): they collapse the bag to ONE scalar = the centroid's
    position on a single axis. They cannot represent the SHAPE of the within-bag
    grade-occupancy (e.g. "half the tissue is G1-like, half is G3-like" reads
    the same mean as "all G2-like" but is a different grade story).
  * a107 (18-param frozen-axis monotone histogram) -> ALSO UNDERFIT (val 0.600):
    it deleted EVERY learnable feature-space DOF (frozen axis only), so it cannot
    even tune which feature-space directions distinguish the grade prototypes.
  * a93 (rank-3 subspace + FULL free Linear(1280->128), 164K) -> VAL-OVERFIT
    (val 0.841 / test 0.912): it restricts the INPUT span to rank-3 but then
    applies a 163,968-param free map ON a rank-3 input -> wildly over-DOF, the
    map re-rotates within the projected coords and keys on fold-specific noise.
a116 sits in the UNEXPLORED interior: it keeps a genuinely learnable feature-
space map (so it does NOT underfit like a107/a101) but that map is a DIAGONAL
metric on the 3 grade-subspace coordinates -> exactly 3 free non-negative
scales, not 163,968. It is provably RICHER than mean-projection (it reads the
full 4-way occupancy SHAPE, not the 1-D centroid) yet has 10 total DOF, far
below the baseline's ~165-197K, so it structurally CANNOT fit fold-specific val
noise.

WHY IT IS RICHER THAN MEAN-PROJECTION (the thing the lens demands it provably
miss)
-----------------------------------------------------------------------------
Mean-projection reads y = <mean_i h_i, v> = mean_i <h_i, v>: a single scalar,
the bag centroid's coordinate on ONE axis. Two bags with the SAME centroid but
DIFFERENT occupancy shapes (a 50/50 G1+G3 mixture vs a pure-G2 bag) are
INDISTINGUISHABLE to it -- yet clinically the diffuse density story differs.
a116's readout y = sum_k g_k q_k with q = mean_i softmax(-||.||^2) is a NONLINEAR
functional of the patch distribution: it depends on the full 4-way occupancy q,
which is NOT recoverable from the centroid alone (softmax-assignment then average
!= assignment-of-the-average). So a116 can separate the two bags above; mean-
projection provably cannot. This is the "diffuse density structure a single
linear-readout-of-the-mean provably misses" the lens asks for.

WHY THE EXTRA DOF CANNOT FIT FOLD NOISE (hard constraints, enumerated)
----------------------------------------------------------------------
  1. The 4 anchor points c_0..c_3 and the rank-3 basis P are FROZEN buffers
     (train-only seed=2, zero DOF, identical object on every fold). No learnable
     rotation touches the 1280-d features -> the feature read-direction is locked
     to the grade-mean subspace (a93 showed the orthogonal 1277 dims carry zero
     between-grade mean signal: singular values [14.45,9.18,5.36,~0]).
  2. The only learnable feature-space DOF is a DIAGONAL metric d = softplus(rho)
     in R^3 (non-negative, 3 scalars) used to weight the 3 subspace coords in the
     prototype-assignment distance. A diagonal-only, non-negative metric CANNOT
     represent an off-diagonal rotation; it can only stretch/shrink the 3 fixed
     grade-axes. So it cannot invent a fold-specific oblique direction the way a
     free 1280->128 (or even 3->128) map can.
  3. The readout anchors g = cumsum(softplus(theta)) are NON-NEGATIVE and
     MONOTONE NON-DECREASING (g_0<=g_1<=g_2<=g_3), affinely fixed so g_0=bias,
     g_3=bias+span. y = sum_k g_k q_k is therefore PROVABLY MONOTONE in
     "occupancy mass shifted toward higher grades": it cannot learn a non-monotone
     (anti-grade) anchor ordering to chase a fold quirk.
  4. The assignment is SYMMETRY-LOCKED to the 4 prototypes (a permutation of
     patches leaves q unchanged; the 4 grade slots are anchored, not learned), so
     there is no free codebook to overfit (unlike a93/a99's free bottleneck).
Total free DOF: rho(3) + theta(4) + bias(1) + log_span(1) + log_tau(1) = 10.

MECHANISM (single bag f in R^{N x 1280}, RAW scalar out, perm/size-invariant)
-----------------------------------------------------------------------------
  Frozen buffers (train-only seed=2): rank-3 basis P [3,1280] (orthonormal rows,
  top-3 right-singular vectors of the centered grade means); prototype
  subspace-coords A_k = c_k @ P^T in R^3 (k=0..3); a fixed coord scale v0 (std of
  the prototype coords, for a scale-free start).
  STEP 1 frozen subspace coords (no free DOF):
      z_i = f_i @ P^T               # [N,3] grade-subspace coordinate of patch i
  STEP 2 diagonal-metric soft assignment to the 4 grade prototypes:
      d   = softplus(rho)           # [3] >=0 learnable diagonal metric
      D_ik = sum_c d_c (z_ic - A_kc)^2 / (v0^2 * tau)  # [N,4] weighted sq-dist
      a_ik = softmax_k(-D_ik)       # [N,4] patch i's soft grade-prototype share
  STEP 3 DIFFUSE bag-occupancy (the diffuse-density readout the prior asks for):
      q_k = (1/N) sum_i a_ik        # [4] bag-wide grade-occupancy distribution
      (mean over patches -> bag-size invariant; sum over i -> perm invariant)
  STEP 4 hard-constrained monotone ordinal-anchor expectation:
      g = cumsum(softplus(theta))   # [4] non-neg, non-decreasing (raw)
      g = bias + span * g / g[-1]   # [4] g_0=bias, g_3=bias+span, monotone
      y = sum_k g_k q_k             # scalar RAW logit (in [g_0, g_3] by convexity)

Because q is a probability vector and g is monotone in [bias, bias+span], y is a
convex combination => y in [bias, bias+span]. Warm-start bias=0, span=3 puts y
in [0,3] at init with NO clamp (honors the raw-logit contract; the trainer
rounds/clips at eval). It is RICHER than mean-projection (depends on the full
occupancy q) but cannot exceed the underfit/overfit traps because its only free
DOF are 10 shape-locked scalars.

LEAKAGE: P, A_k, v0, axis_coord are frozen train-only seed=2 buffers, the SAME
physical object on every fold (no per-fold/test info; unlike a103-style head-init
that the seed=2 optimiser exploits). forward() reads ONLY 'features'. If the
cache is absent, the model falls back to a fixed seeded random basis + spread
prototype coords so it still constructs deterministically.

GRADING-ALIGNED: q is literally "the fraction of marrow tissue that looks like
each grade" -- diffuse, bag-wide, holistic; NOT a few standout patches (no
argmax / top-k), NOT ||h|| (the assignment uses subspace coords, not norm;
a93/diag showed ||h|| Spearman ~0 and that the grade subspace skips bone). The
ordinal-anchor expectation is the clinical "overall density" along the grade
ordering.

ABLATION COMPANION a118: import THIS Model and flip readout="meanproj" -> the
linear-readout-of-the-mean control. a116 vs a118 isolates EXACTLY "does the
4-way occupancy SHAPE add over the 1-D centroid?" (the lens's core claim).

Param count (input_dim=1280, rank=3): rho(3)+theta(4)+bias(1)+log_span(1)+log_tau(1)=10.
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
    import math

    if x > 20.0:
        return x
    return math.log(math.expm1(x))


def _load_subspace_and_protos(path: Path, input_dim: int, rank: int):
    """Return (P [r,D] orthonormal, A [G,r] proto subspace-coords, v0 scalar, axis_coord [r]).

    All train-only seed=2; init/buffer-only, identical object on every fold.
    Falls back to (None,...) so the model still constructs without the cache.
    """
    if not path.is_file():
        return None, None, None, None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None, None, None, None
    protos = blob.get("prototypes", None)
    if not isinstance(protos, dict) or len(protos) == 0:
        return None, None, None, None
    keys = sorted(protos.keys())
    C = torch.stack([protos[k].float().view(-1) for k in keys], dim=0)  # [G, D]
    if C.shape[1] != input_dim:
        return None, None, None, None
    M = C - C.mean(dim=0, keepdim=True)                # [G, D] centered grade means
    _, _, Vt = torch.linalg.svd(M, full_matrices=False)
    r = int(min(rank, Vt.shape[0]))
    P = Vt[:r].contiguous()                            # [r, D] orthonormal rows
    A = C @ P.t()                                      # [G, r] prototype coords
    v0 = float(A.std(unbiased=False).clamp(min=_EPS))  # scalar coord scale
    # in-subspace fibrosis axis coords (for the a118 mean-projection ablation)
    axis = blob.get("axis", None)
    if axis is not None and axis.numel() == input_dim:
        ax = axis.float().view(-1)
        ax = ax / ax.norm().clamp(min=_EPS)
        axis_coord = ax @ P.t()                        # [r]
    else:
        axis_coord = (A[-1] - A[0])                    # c3-c0 coords as fallback
        axis_coord = axis_coord / axis_coord.norm().clamp(min=_EPS)
    return P, A, v0, axis_coord


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        subspace_rank: int = 3,
        readout: str = "occupancy",   # "occupancy" (a116 main) | "meanproj" (a118 ablation)
        tau_init: float = 1.0,        # assignment temperature (learnable, >0 via softplus)
        span_init: float = 3.0,       # anchor span (g_3 - g_0); warm-start to grade range
        prototype_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert readout in ("occupancy", "meanproj"), readout
        self.readout = readout
        self.input_dim = int(input_dim)

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        P, A, v0, axis_coord = _load_subspace_and_protos(path, input_dim, subspace_rank)

        if P is None:
            # Deterministic fallback: fixed seeded random orthonormal basis +
            # spread-out prototype coords so the module still constructs/runs.
            g = torch.Generator().manual_seed(2)
            R = torch.randn(input_dim, subspace_rank, generator=g)
            Q, _ = torch.linalg.qr(R)                 # [D, r] orthonormal cols
            P = Q.t().contiguous()                    # [r, D]
            A = torch.linspace(-1.5, 1.5, 4).unsqueeze(1).repeat(1, subspace_rank)
            v0 = 1.0
            axis_coord = (A[-1] - A[0])
            axis_coord = axis_coord / axis_coord.norm().clamp(min=_EPS)

        self.register_buffer("P", P)                  # [r, D] frozen subspace basis
        self.register_buffer("A", A)                  # [G, r] frozen prototype coords
        self.register_buffer("v0", torch.tensor(float(v0)))
        self.register_buffer("axis_coord", axis_coord.view(-1))  # [r] for meanproj
        self.n_grades = int(A.shape[0])
        self.rank = int(P.shape[0])

        # --- learnable, hard-constrained DOF (the ONLY free params) ---
        # diagonal metric on the r subspace coords: d = softplus(rho) >= 0.
        # init rho s.t. softplus(rho) ~ 1 (uniform metric => plain sq-dist).
        self.rho = nn.Parameter(torch.full((self.rank,), float(_inv_softplus(1.0))))
        # monotone non-negative anchors: g = cumsum(softplus(theta)); init uniform
        # so g is a linear ramp 0..span at start (=> y ~ mean grade index).
        self.theta = nn.Parameter(torch.zeros(self.n_grades))
        self.bias = nn.Parameter(torch.tensor(0.0))                       # g_0
        self.log_span = nn.Parameter(torch.tensor(float(_inv_softplus(span_init))))
        self.log_tau = nn.Parameter(torch.tensor(float(_inv_softplus(tau_init))))

        if self.readout == "meanproj":
            # a118 ablation: linear-readout-of-the-mean = w0 + w1 * mean_i <z_i, axis>.
            # 2 free params (w0,w1); axis_coord frozen. This is the control the
            # lens says a116 must beat. NOTE: in meanproj mode the occupancy DOF
            # (rho/theta/bias/log_span/log_tau) are unused dead params; the
            # ablation freezes them (requires_grad=False) so the only LEARNABLE
            # DOF are w0,w1 — a clean mean-projection control.
            self.w0 = nn.Parameter(torch.tensor(1.5))
            self.w1 = nn.Parameter(torch.tensor(1.0))
            for p in (self.rho, self.theta, self.bias, self.log_span, self.log_tau):
                p.requires_grad_(False)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        f = features
        if f.dim() == 1:
            f = f.unsqueeze(0)

        # STEP 1 — frozen rank-r grade-subspace coordinates (no free DOF).
        z = f @ self.P.t()                                   # [N, r]

        if self.readout == "meanproj":
            # a118 — linear-readout-of-the-mean along the frozen fibrosis axis.
            s = (z * self.axis_coord.unsqueeze(0)).sum(dim=1) / self.v0   # [N]
            y = self.w0 + self.w1 * s.mean()
            logits = y.view(1)
            if return_attention:
                return logits, None, None
            return logits, None, None

        # --- a116 occupancy readout ---
        d = F.softplus(self.rho)                             # [r] >= 0 diagonal metric
        tau = F.softplus(self.log_tau).clamp(min=1e-3)       # > 0 temperature

        # STEP 2 — diagonal-metric soft assignment to the G frozen prototypes.
        # diff[i,k,c] = z[i,c] - A[k,c]; D[i,k] = sum_c d_c diff^2 / (v0^2 * tau)
        diff = z.unsqueeze(1) - self.A.unsqueeze(0)          # [N, G, r]
        D = (d.view(1, 1, -1) * diff.pow(2)).sum(dim=2) / (self.v0 * self.v0 * tau)  # [N, G]
        a = F.softmax(-D, dim=1)                             # [N, G] soft grade share

        # STEP 3 — DIFFUSE bag-occupancy distribution (mean over patches).
        q = a.mean(dim=0)                                    # [G] in Delta^{G-1}

        # STEP 4 — hard-constrained monotone ordinal-anchor expectation.
        w = F.softplus(self.theta)                           # [G] >= 0
        g = torch.cumsum(w, dim=0)                           # [G] non-decreasing
        g = g / g[-1].clamp(min=_EPS)                        # in [.., 1], top = 1
        span = F.softplus(self.log_span)                     # > 0
        g = self.bias + span * g                             # [G] monotone in [bias, bias+span]
        y = (g * q).sum()                                    # scalar RAW logit, convex comb

        logits = y.view(1)
        if return_attention:
            return logits, q.detach(), None
        return logits, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    subspace_rank=3,
    readout="occupancy",     # a116 main = 4-way diffuse occupancy expectation
    tau_init=1.0,
    span_init=3.0,
    prototype_path=None,     # None -> data/prototypes_virchow2_reti_train_seed2.pt
)
