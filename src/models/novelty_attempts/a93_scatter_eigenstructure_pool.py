"""a93 — within-bag SCATTER EIGEN-STRUCTURE pool (multivariate 2nd-moment /
set-GEOMETRY signature of the patch cloud).

LENS (set-GEOMETRY / 2nd-moment)
--------------------------------
Summarise the patch cloud by the SHAPE of its within-bag SCATTER (the
covariance eigen-structure) inside a small fibrosis-aligned subspace, read as a
robust DIFFUSE-density signature. The grade-evidence is "how the cloud spreads",
not "where the mean sits" and not "the variance of one 1-D projection". The
genuinely new ingredient is the MULTIVARIATE within-bag covariance restricted to
a K-dim subspace whose first axis is the warm-started fibrosis direction:

    - the TRACE of the scatter (total diffuse spread across the subspace),
    - the spread ALONG the fibrosis axis vs OFF it (an ANISOTROPY ratio), and
    - the CROSS-covariance energy coupling the fibrosis axis to its
      neighbourhood (an off-DIAGONAL covariance term — by construction NOT a
      function of any single 1-D moment).

Why this is DISTINCT from a66 / a64 (mean/var/skew of a 1-D projection)
-----------------------------------------------------------------------
a64/a66 reduce the bag to ONE scalar projection s_i = <f_i, axis> and read
central moments of that single number (var, skew). Those are diagonal entries of
a 1x1 covariance — the (0,0) cell only. a93 forms the FULL K x K within-bag
covariance C of the cloud in a K-dim subspace and reads its EIGEN-STRUCTURE:
the OFF-diagonal cross-covariances C[0, 1:] and the off-axis trace tr(C) - C[0,0]
are quantities that DO NOT EXIST in a 1-D projection and CANNOT be recovered from
m1/m2/m3 of <f_i, axis>. The anisotropy ratio along/tr is a multivariate SHAPE
descriptor (how elongated the cloud is along the fibrosis direction relative to
its total spread), not a 1-D moment. So a93 is orthogonal to a64/a66 by
construction: removing the multivariate terms (the ablation a94) lands EXACTLY
on a single-projection variance readout.

Mechanism (exact forward-pass math; single bag, features f in R^{N x 1280})
---------------------------------------------------------------------------
  Q in R^{1280 x K}            # FROZEN orthonormal subspace basis (buffer, no grad).
                               # Column 0 = warm-started train-only fibrosis axis
                               # (seed=2). Columns 1..K-1 = deterministic random
                               # directions, Gram-Schmidt-orthonormalised against
                               # the axis. K small (default 8).
  P    = f @ Q                 # [N, K]  cloud coordinates in the subspace (LINEAR)
  mu   = mean_i P_i            # [K]     subspace centroid
  Pc   = P - mu                # [N, K]  centred cloud
  C    = (Pc^T @ Pc) / N       # [K, K]  WITHIN-BAG COVARIANCE (biased, 2nd moment)
  tr     = sum_k C[k,k]                      # total diffuse spread (scatter trace)
  along  = C[0,0]                            # spread ALONG the fibrosis axis
  off    = tr - along                        # spread OFF the fibrosis axis
  aniso  = along / (tr + eps)                # fraction of scatter on the axis (SHAPE)
  cross  = || C[0, 1:] ||_2                  # axis<->neighbourhood cross-cov energy
  g      = [ log1p(tr), log1p(along), log1p(off), aniso, log1p(cross) ]   # 5-vec
  y      = Linear(5 -> 1)(g)                 # tiny RAW readout (clamp only at eval, by trainer)

log1p compresses the (positive) variance magnitudes for scale stability at the
real Virchow2 feature scale (~6) and keeps the 5 descriptors O(1) so the head
stays well-conditioned across folds. eps guards the ratio. All five descriptors
are symmetric means over patches (covariance is a mean of outer products), so
they are PERMUTATION-INVARIANT and BAG-SIZE-INVARIANT; the biased 1/N covariance
is exactly invariant to bag duplication. N==1 -> C=0 -> g=[0,0,0,0,0] (safe).

The ONLY learnable parameters are the 5-in linear head (5 weights + 1 bias = 6).
Q is a frozen buffer (geometry is a fixed, deterministic function of the input),
so the model has essentially ZERO per-seed capacity to memorise the 214-ROI
cohort — the active ingredient is a fixed multivariate STATISTIC, and only its
linear combination is fit. This is the deliberate variance-reduction stance: the
documented failure mode is val<->test collapse from over-fitting a tiny cohort,
so a93 spends its degrees of freedom on a 6-param readout of a robust,
fold-stable geometric signature rather than on a fold-fragile high-capacity MLP.

Why NOT each tried mechanism (orthogonality audit)
--------------------------------------------------
  - NOT a64/a66 (1-D moments): a93 reads OFF-diagonal covariance (cross) and the
    OFF-axis trace (off) — multivariate scatter terms that are identically zero
    in / absent from a single 1-D projection's moments. The ablation a94 (K=1)
    is precisely a single-axis variance readout, and a93 (K=8) vs a94 isolates
    EXACTLY the multivariate eigen-structure.
  - NOT mean-pool / a53 / a57: the centroid mu is CENTRED OUT (C uses Pc = P-mu);
    g contains NO mean term at all. The readout is a pure 2nd-moment function;
    it carries zero first-order (mean-density) signal, so it cannot collapse to
    mean-pool. (a93 is mean-FREE by construction.)
  - NOT a72/a73 (pairwise Gini-spread): a72 is a 1-D mean-absolute-pairwise-
    difference of <f_i, axis> (still scalar, still 1 axis). a93 is the K x K
    covariance MATRIX and its eigen-/anisotropy structure — a different functional
    in a multi-D subspace.
  - NOT coverage / threshold / quantile / dist-match: no count above tau, no
    quantile, no reference distribution; only raw covariance invariants.
  - NOT gated/PMA/top-k/rank/consensus/graph-smooth: NO per-patch scorer, NO
    selection, NO attention, NO re-weighting, NO propagation. Every patch enters
    the covariance symmetrically.
  - NOT ||h||-norm salience: C is computed from CENTRED subspace coordinates; raw
    feature norm ||f_i|| is never used to weight anything.
  - NOT heavy-dropout / input-noise / ensemble / ordinal-link: deterministic
    closed-form geometry + a single linear head; no stochastic regulariser, no
    cumulative-link, no multi-mechanism blend.

PAIRED cross-fold robustness argument (the real bar, NOT a seed=2 lottery)
--------------------------------------------------------------------------
  1. Near-zero per-seed capacity: 6 learnable params total. There is no
     high-capacity block to overfit a fold, which is the documented root cause of
     the val<->test gap (a84 cleared seed=2 yet lost 4/5 folds because it had a
     fold-fragile learnable smoothing + bottleneck). a93's signature is the SAME
     deterministic function of features in every fold; only a 6-param line is fit,
     so the val<->test gap is structurally small.
  2. Covariance is a LOW-VARIANCE bag statistic: a mean of outer products over
     all N patches (typically tens-hundreds), so it is smooth and stable across
     resamples of the cohort — it does not chase individual patches the way top-K
     / attention does, and it does not hinge on a single fragile threshold.
  3. The subspace is FIXED (frozen Q) and grounded: column 0 is the +0.84
     held-out fibrosis axis; the off-axis directions are a fixed orthonormal frame
     shared across folds, so the geometric descriptors mean the same thing in
     every fold. Nothing about the geometry is re-fit per seed, so a seed=2 win
     cannot be a lottery — the only fitted object is the 6-param readout, which is
     too small to memorise a fold.
  4. Multivariate SHAPE is the hypothesised fold-stable invariant of diffuse
     fibrosis: a truly diffuse high-grade marrow has a cloud that is broadly and
     ISOTROPICALLY spread (high tr, low anisotropy, high off-axis spread), whereas
     a focal low-grade bag with a few hot tiles has a cloud ELONGATED along the
     fibrosis axis (anisotropy high, cross high) at a similar mean. This shape
     contrast is a population-level geometric property that should transfer
     across folds better than a mean threshold.

Ablation companion: a94_scatter_eigenstructure_pool_k1.py — sets subspace_dim=1
(K=1). Then Q = [axis] only, C is 1x1, so g = [log1p(var), log1p(var), 0, 1, 0]:
the readout has access ONLY to the single-axis variance (a 1-D 2nd moment, i.e.
the a64 spread term) and a constant. This removes EXACTLY the active ingredient —
the MULTIVARIATE eigen-structure (off-axis spread, anisotropy, cross-covariance)
— while keeping the identical pipeline (frozen axis, centred covariance, log1p,
6-param-shaped head). a93 (K=8) vs a94 (K=1) therefore isolates "does the
multivariate scatter shape beat the single-axis variance?".

Kill criterion: abandon if a93 does not improve the PAIRED cross-fold Δ (val AND
test) over the gated-attention baseline on a majority of folds {0,1,2,3,42}.
seed=2 alone is NOT sufficient (the a84 lesson).

Param count (input_dim=1280, K=8): Q is a FROZEN buffer (0 trainable). Learnable
= head Linear(5,1) weight 5 + bias 1 = 6 trainable parameters (Q buffer占 1280*8
= 10,240 stored floats, all frozen). Well under the 197K cap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis (1280-d) from the train-only prototype cache (seed=2).

    Returns None if the cache is absent (the model then uses a deterministic
    random first axis so it still constructs and the trainer never breaks). Uses
    ONLY the train-derived axis — no test information, init-only.
    """
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    v = blob["axis"].float().view(-1)
    if v.numel() != input_dim:
        return None
    return v / v.norm().clamp(min=1e-8)


def _build_subspace(axis: Optional[torch.Tensor], input_dim: int, k: int, seed: int) -> torch.Tensor:
    """Orthonormal basis Q [input_dim, k] with column 0 = fibrosis axis.

    Columns 1..k-1 are deterministic random directions, made orthonormal to the
    axis (and each other) by QR. Fully reproducible from `seed`, identical across
    folds -> the geometric descriptors mean the same thing in every fold.
    """
    g = torch.Generator().manual_seed(int(seed))
    W = torch.randn(input_dim, k, generator=g)
    if axis is not None:
        W[:, 0] = axis  # plant the fibrosis axis as the seed of the frame
    # QR gives an orthonormal basis; force column 0 to align with the axis sign.
    Q, R = torch.linalg.qr(W)
    if axis is not None:
        # qr can flip signs; realign column 0 to the fibrosis axis direction.
        if (Q[:, 0] @ axis) < 0:
            Q[:, 0] = -Q[:, 0]
    return Q  # [input_dim, k], Q^T Q = I_k


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        subspace_dim: int = 8,          # K: a93 main = 8 (multivariate); a94 ablation = 1
        warm_start: bool = True,        # column 0 of Q = train fibrosis axis (seed=2)
        prototype_path: Optional[str] = None,
        frame_seed: int = 2,            # deterministic off-axis frame (init-only)
        eps: float = 1e-4,
        clamp_output: bool = False,     # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert subspace_dim >= 1, "subspace_dim (K) must be >= 1."
        self.K = int(subspace_dim)
        self.eps = float(eps)
        self.clamp_output = bool(clamp_output)

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        axis = _load_fibrosis_axis(path, input_dim) if warm_start else None
        Q = _build_subspace(axis, input_dim, self.K, frame_seed)

        # FROZEN orthonormal subspace basis (no gradient): the geometry is a
        # fixed deterministic function of the input. Stored as a buffer so it
        # moves with the module device.
        self.register_buffer("Q", Q)  # [input_dim, K]

        # Number of geometric descriptors fed to the head:
        #   [log1p(tr), log1p(along), log1p(off), aniso, log1p(cross)] = 5.
        self.n_desc = 5
        self.head = nn.Linear(self.n_desc, num_classes)
        with torch.no_grad():
            # Warm-start the readout so the model STARTS as a monotone reader of
            # the diffuse spread (more total/along-axis scatter -> higher grade)
            # and must EARN the multivariate shape terms. Small positive slope on
            # the scatter-magnitude descriptors, ~0 on the pure shape ratios,
            # mid-grade bias. This makes a93 start close to its a94 ablation so
            # the comparison has no capacity confound.
            self.head.weight.zero_()
            self.head.weight[0, 0] = 0.20   # log1p(tr)    : total diffuse spread
            self.head.weight[0, 1] = 0.20   # log1p(along) : spread along axis
            self.head.bias.fill_(1.5)       # interior of [0, 3]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] -- ONE bag (trainer contract, batch_size=1).
        P = features @ self.Q                       # [N, K] cloud coords in subspace
        n = P.size(0)

        if n > 1:
            mu = P.mean(dim=0, keepdim=True)         # [1, K] subspace centroid
            Pc = P - mu                              # [N, K] CENTRED cloud (mean removed)
            C = (Pc.t() @ Pc) / n                    # [K, K] within-bag covariance (biased)
        else:
            C = torch.zeros(self.K, self.K, device=P.device, dtype=P.dtype)

        diag = torch.diagonal(C)                     # [K] per-direction variances
        tr = diag.sum()                              # total diffuse spread (trace)
        along = diag[0]                              # spread ALONG the fibrosis axis
        off = (tr - along).clamp(min=0.0)            # spread OFF the fibrosis axis
        aniso = along / (tr + self.eps)              # SHAPE: fraction of scatter on axis
        if self.K > 1:
            cross = C[0, 1:].pow(2).sum().sqrt()     # axis<->neighbourhood cross-cov energy
        else:
            cross = torch.zeros((), device=P.device, dtype=P.dtype)

        g = torch.stack([
            torch.log1p(tr),
            torch.log1p(along),
            torch.log1p(off),
            aniso,
            torch.log1p(cross),
        ]).view(1, self.n_desc)                       # [1, 5]

        y = self.head(g).view(-1)                     # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # No per-patch saliency in this design (the signal is bag-geometry);
            # expose the per-patch axis coordinate as a read-only diagnostic.
            return y, P[:, 0].detach(), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    subspace_dim=8,        # a93 main = multivariate scatter eigen-structure (K=8)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    frame_seed=2,
    eps=1e-4,
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
