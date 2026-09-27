"""a97 — MEAN-FREE multivariate covariance-SHAPE readout (round-4).

LENS (set-GEOMETRY / 2nd-moment, mean-removed)
----------------------------------------------
Read the grade from the SHAPE of the within-bag SCATTER (covariance
eigen-structure) of the patch cloud inside a small, frozen, fibrosis-aligned
subspace. The evidence is "how the cloud SPREADS", not "where its mean sits"
and not "the variance of one 1-D projection". The centroid is REMOVED before
the covariance is formed, so the descriptor carries ZERO first-order
(mean-density) signal: the design is MEAN-FREE by construction and therefore
*cannot* collapse to mean-pool (the recurring a56/a57/a62 trap).

The genuinely new ingredient over a 1-D moment readout (a64/a66) is the
MULTIVARIATE within-bag covariance restricted to a K-dim subspace whose first
axis is the warm-started train-only fibrosis direction:

    - the TRACE of the scatter            (total diffuse spread across K dirs),
    - the spread ALONG the fibrosis axis vs OFF it (an ANISOTROPY ratio), and
    - the CROSS-covariance energy coupling the fibrosis axis to its off-axis
      neighbourhood (an OFF-diagonal covariance term — by construction NOT a
      function of any single 1-D moment).

Why DISTINCT from a64/a66 (mean / var / skew of a 1-D projection)
-----------------------------------------------------------------
a64/a66 reduce the bag to ONE scalar projection s_i = <f_i, axis> and read
central moments of that single number. Those are the (0,0) cell of a 1x1
covariance. a97 forms the FULL K x K within-bag covariance C of the cloud and
reads its eigen-structure: the OFF-diagonal cross-covariance ||C[0,1:]||_2 and
the OFF-axis trace tr(C) - C[0,0] are quantities that DO NOT EXIST in a 1-D
projection and CANNOT be recovered from m1/m2/m3 of <f_i, axis>. The anisotropy
ratio along/tr is a multivariate SHAPE descriptor (how elongated the cloud is
along the fibrosis direction relative to its total spread), not a 1-D moment.
Removing the multivariate terms (the ablation a98, K=1) lands EXACTLY on a
single-axis variance readout — so a97 (K=8) vs a98 (K=1) isolates EXACTLY the
multivariate covariance SHAPE.

GRADING RATIONALE (advisor's diffuse-density prior)
---------------------------------------------------
Grade = OVERALL / DIFFUSE density of the reticulin meshwork, NOT a property of a
few patches and NOT ||h||-weighted (feature-norm tracks tissue-vs-background,
Spearman ~0 with grade, a nuisance). A truly DIFFUSE high-grade marrow has a
cloud that is broadly and roughly ISOTROPICALLY spread (high tr, low anisotropy,
high off-axis spread); a FOCAL low-grade bag with a few hot tiles has a cloud
ELONGATED along the fibrosis axis (high anisotropy, high cross) at a similar
mean. This shape contrast is a population-level geometric property, hypothesised
to transfer across folds better than a mean threshold.

Mechanism (exact forward-pass math; single bag, features f in R^{N x 1280})
---------------------------------------------------------------------------
  Q in R^{1280 x K}            # FROZEN orthonormal subspace basis (buffer, no grad).
                               # Column 0 = warm-started train-only fibrosis axis
                               # (data/prototypes_virchow2_reti_train_seed2.pt['axis'],
                               # seed=2, train patients only). Columns 1..K-1 are
                               # deterministic random directions, QR-orthonormalised
                               # against the axis (fixed frame_seed=2), identical in
                               # every fold so the descriptors mean the same thing.
  P    = f @ Q                 # [N, K]  cloud coordinates in the subspace (LINEAR)
  mu   = mean_i P_i            # [K]     subspace centroid
  Pc   = P - mu                # [N, K]  CENTRED cloud (mean REMOVED -> mean-free)
  C    = (Pc^T @ Pc) / N       # [K, K]  WITHIN-BAG COVARIANCE (biased, 2nd moment)
  tr     = sum_k C[k,k]                      # total diffuse spread (scatter trace)
  along  = C[0,0]                            # spread ALONG the fibrosis axis
  off    = tr - along                        # spread OFF the fibrosis axis
  aniso  = along / (tr + eps)                # fraction of scatter on the axis (SHAPE)
  cross  = || C[0, 1:] ||_2                  # axis<->neighbourhood cross-cov energy
  g      = [ log1p(tr), log1p(along), log1p(off), aniso, log1p(cross) ]   # 5-vec
  y      = Linear(5 -> 1)(g)                 # tiny RAW readout (trainer rounds/clips at eval)

log1p compresses the (positive) variance magnitudes for scale stability at the
real Virchow2 feature scale (~6) and keeps the descriptors O(1) so the head
stays well-conditioned across folds. eps guards the ratio. All five descriptors
are symmetric means over patches (covariance is a mean of outer products), so
they are PERMUTATION-INVARIANT and BAG-SIZE-INVARIANT (the biased 1/N covariance
is exactly invariant to bag duplication). N==1 -> C=0 -> g=[0,0,0,0,0] -> y=bias
(the warm-start bias, a safe mid-grade value).

DETERMINISM (critical) — no eigendecomposition / no power iteration is used.
The descriptors are read directly from the closed-form covariance entries
(trace, a diagonal cell, a row-slice norm). There is NO torch.linalg.eig /
svd / power-iteration anywhere, so there is no sign-ambiguity or iterative
nondeterminism: two eval passes on the same bag are bit-identical.

WHY NOT each tried mechanism (orthogonality audit)
--------------------------------------------------
  - NOT mean-pool / a53 / a57: centroid mu is CENTRED OUT; g has NO mean term.
    Pure 2nd-moment function -> cannot collapse to mean-pool (MEAN-FREE).
  - NOT a64/a66 (1-D moments): reads OFF-diagonal cross-cov and OFF-axis trace,
    absent from a single 1-D projection; ablation a98 (K=1) is precisely the
    1-D variance readout.
  - NOT a72/a73 (pairwise Gini-spread): a72 is a 1-D mean-abs-pairwise-diff of
    <f_i, axis>; a97 is the K x K covariance MATRIX and its anisotropy/cross
    structure in a multi-D subspace.
  - NOT coverage / threshold / quantile / dist-match: no count above tau, no
    quantile, no reference distribution; only raw covariance invariants.
  - NOT gated / PMA / top-k / rank / consensus / graph-smooth: NO per-patch
    scorer, NO selection, NO attention, NO re-weighting, NO propagation. Every
    patch enters the covariance symmetrically.
  - NOT ||h||-norm salience: C is from CENTRED subspace coords; raw feature norm
    never weights anything.

Warm-start (init-only, train-derived, no test leakage)
------------------------------------------------------
Column 0 of Q is the train-only (seed=2) fibrosis axis. The head is warm-started
to read the scatter MAGNITUDE monotonically (small positive slope on log1p(tr)
and log1p(along), zero on the pure shape ratios, mid-grade bias 1.5) so a97
STARTS close to its a98 ablation and must EARN the multivariate shape terms —
no capacity confound in the comparison. With N==1 the model returns exactly the
bias (1.5), a safe interior value.

Param count (input_dim=1280, K=8)
---------------------------------
  Q : FROZEN buffer (1280*8 = 10,240 stored floats, 0 trainable).
  head Linear(5 -> 1) : weight 5 + bias 1 = 6 trainable parameters.
  TOTAL trainable = 6  (<< the ~197K cap; near-zero per-seed capacity by design).

Ablation companion: a98_scatter_eigenstructure_ablation.py imports THIS Model and
sets subspace_dim=1 (K=1). Then Q = [axis] only, C is 1x1, and g collapses to
[log1p(var), log1p(var), 0, ~1, 0] — the head reads ONLY the single-axis
variance (the a64 spread term) + constants. a97 (K=8) vs a98 (K=1) isolates
EXACTLY: "does the multivariate covariance SHAPE beat a 1-D variance?".
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
    Q, _ = torch.linalg.qr(W)
    if axis is not None and (Q[:, 0] @ axis) < 0:
        # QR can flip signs; realign column 0 to the fibrosis axis direction so
        # `along` is read off the grounded direction (sign is irrelevant to the
        # squared covariance, but we keep it explicit and deterministic).
        Q[:, 0] = -Q[:, 0]
    return Q  # [input_dim, k], Q^T Q = I_k


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        subspace_dim: int = 8,          # K: a97 main = 8 (multivariate); a98 ablation = 1
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

        # FROZEN orthonormal subspace basis (no gradient): the geometry is a fixed
        # deterministic function of the input. Stored as a buffer so it moves with
        # the module device and is not counted as a trainable parameter.
        self.register_buffer("Q", Q)  # [input_dim, K]

        # Number of geometric descriptors fed to the head:
        #   [log1p(tr), log1p(along), log1p(off), aniso, log1p(cross)] = 5.
        self.n_desc = 5
        self.head = nn.Linear(self.n_desc, num_classes)
        with torch.no_grad():
            # Warm-start the readout so the model STARTS as a monotone reader of
            # the diffuse spread (more total / along-axis scatter -> higher grade)
            # and must EARN the multivariate shape terms. Small positive slope on
            # the scatter-magnitude descriptors, ~0 on the pure shape ratios,
            # mid-grade bias. This makes a97 start close to its a98 ablation so the
            # comparison has no capacity confound. N==1 -> g=0 -> y=bias (1.5).
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
            # N==1 -> no spread defined -> zero covariance -> g=0 -> y=bias (safe).
            C = torch.zeros(self.K, self.K, device=P.device, dtype=P.dtype)

        diag = torch.diagonal(C)                     # [K] per-direction variances
        tr = diag.sum()                              # total diffuse spread (trace)
        along = diag[0]                              # spread ALONG the fibrosis axis
        off = (tr - along).clamp(min=0.0)            # spread OFF the fibrosis axis
        aniso = along / (tr + self.eps)              # SHAPE: fraction of scatter on axis
        if self.K > 1:
            cross = C[0, 1:].pow(2).sum().clamp(min=0.0).sqrt()  # axis<->off-axis cross-cov energy
        else:
            cross = torch.zeros((), device=P.device, dtype=P.dtype)

        g = torch.stack([
            torch.log1p(tr.clamp(min=0.0)),
            torch.log1p(along.clamp(min=0.0)),
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
    subspace_dim=8,        # a97 main = multivariate scatter eigen-structure (K=8)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    frame_seed=2,
    eps=1e-4,
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
