"""a93 — Project onto a low-rank GRADE-RELEVANT SUBSPACE, then diffuse-pool.

LENS = feature-SUBSPACE. This changes the *representation*, NOT the pool. The
pool stays the plainest possible diffuse mean. The active ingredient is a
fixed, train-only rank-r=3 subspace of the raw 1280-d Virchow2 space onto
which every patch feature is projected (orthogonal-component dropped) BEFORE the
bottleneck. The 1280 - r grade-IRRELEVANT directions never reach the encoder, so
the model cannot overfit on them.

Why a rank-3 subspace is the *right* object here (data-driven, train-only)
--------------------------------------------------------------------------
The cached train-only prototypes (seed=2) give 4 grade-mean vectors
c_0..c_3 in R^1280. After centering by their mean, the matrix
[c_0-cbar; ...; c_3-cbar] has rank EXACTLY 3 (verified: singular values
[14.45, 9.18, 5.35, 3.8e-6]). That 3-D span is, by construction, the ONLY
subspace in which the per-grade population means differ — i.e. the linear
"grade-relevant subspace" the lens asks for, with k=3 << 128 << 1280. Anything
orthogonal to it carries ZERO between-grade mean signal (only nuisance: stain,
tissue-vs-background, scanner). a25/a26 (DE22) showed a SINGLE such axis
(c_3-c_0) is too narrow to *score* patches; the open follow-up the post-mortem
named verbatim was "multi-axis ... rank-K projection prior ... still untested".
a93 tests it, but as a REPRESENTATION restriction (project the feature), not as
an attention scorer (the DE22 failure mode).

Mechanism (exact forward math)
------------------------------
Let P in R^{r x 1280} be the FROZEN orthonormal basis of the rank-3
grade-subspace (built once at init from the train prototypes, then a buffer):
    M    = stack(c_0,...,c_3) - mean_g c_g            # [4,1280], centered means
    P    = top-r right-singular vectors of M          # [r,1280], orthonormal rows
    (r=3; P P^T = I_r.)

Forward (single bag features f in R^{N x 1280}):
    coords = f @ P^T                                  # [N, r]   grade-subspace coords
    fproj  = coords @ P                               # [N,1280] = f restricted to span(P)
                                                      #   (orthogonal-complement DROPPED)
    h_i    = Dropout(ReLU(W1 fproj_i + b1))           # [N,128]  baseline encoder
    z      = mean_i h_i                               # [128]    DIFFUSE pool (plain mean)
    y      = W2 z + b2                                # [1]      RAW grade logit

The pool is the unweighted mean — IDENTICAL to a92/a88 (no attention, no
selection, no salience, no graph step). The ONLY change vs the diffuse baseline
is the term `fproj = f P^T P` inserted before the bottleneck: a fixed rank-3
orthogonal projector applied to the raw feature.

Warm-start (train-only, init-only) — matches a83 idiom
------------------------------------------------------
Because the encoder now only ever sees vectors inside span(P), we point the
linear head at the in-subspace fibrosis response. The fibrosis axis
data/prototypes_..._seed2.pt['axis'] = (c_3-c_0)/||.|| LIES in span(P) (it is a
difference of grade means), so projecting it changes nothing; we compute the
bottleneck response W1 (P^T P axis) and copy a scaled, unit-normed version into
the head weight, bias=1.5 (mid-grade). Pure initialisation; fully learned after.

Why this is genuinely ORTHOGONAL to every tried mechanism
---------------------------------------------------------
- It is NOT a pooling change. The pool is the plain diffuse mean used by the
  TIED baselines a88/a92. mean-pool, top-k, rank/norm-salience, per-patch
  severity, consensus, contiguity, PMA, soft-rank, moments, quantiles,
  pairwise-spread, dist-match, graph-smoothing, robust-trimmed, ordinal head,
  ensembles — ALL of those modify HOW patches are combined or HOW the scalar is
  read out. a93 leaves combination + readout untouched and instead restricts the
  per-patch REPRESENTATION to a low-rank subspace. No prior attempt drops the
  orthogonal complement of a grade-subspace before pooling.
- It is NOT DE22 single-axis projection. DE22 used <f, axis> as a per-patch
  ATTENTION SCORER (a softmax weight) — it changed the pool and collapsed each
  patch to one scalar. a93 keeps the FULL 3-D subspace coordinates (then the
  128-d bottleneck), changes NO weight, and feeds a plain mean. A single axis
  cannot span the rank-3 grade signal; a93 keeps all 3 mean-difference
  directions.
- It is NOT input dropout (DE19). Dropout zeros RANDOM dims with train-time
  noise (stochastic, inference-identical to no-op). a93 deterministically keeps
  a FIXED, data-chosen, low-rank subspace at BOTH train and inference. It is a
  structural representation prior, not regularisation noise. (a93's own dropout
  is the baseline p=0.5, unchanged from a88/a92 — not the active ingredient.)
- It is NOT L2-norm / norm-salience: nothing is rescaled by ||.||; the projector
  is orthogonal (preserves the in-subspace geometry, just kills out-of-subspace
  variance).

Why it should help PAIRED cross-fold generalisation (not a seed=2 lottery)
--------------------------------------------------------------------------
The 1280-d Virchow2 features carry far more nuisance variation (stain, scanner,
tissue density) than grade signal; with only 30 train patients the bottleneck
W1 (1280->128) is free to fit fold-specific directions that have nothing to do
with grade, which is exactly the val<->test gap this session keeps hitting.
By hard-projecting onto the rank-3 between-grade-mean subspace FIRST, the
encoder's effective input dimensionality drops from 1280 to 3 — the bottleneck
literally cannot key on a fold-specific nuisance direction, because that
direction is zeroed before it. This is a capacity/variance reduction targeted at
generalisation, not at fitting seed=2. The subspace is fixed (train-only,
seed=2) so it injects no per-fold/test info; the SAME projector is reused across
all folds, so any gain is a structural-prior gain, not a seed-2 coincidence.
Ablation (a94) flips the projector to IDENTITY (full 1280-d) with byte-identical
encoder/head/warm-start, isolating EXACTLY "does the rank-3 grade-subspace
restriction help?".

Permutation- & bag-size-invariance
-----------------------------------
P is fixed and applied per-patch (f @ P^T @ P is row-wise), so it is
permutation-EQUIVARIANT; the final mean over patches is permutation-INVARIANT
and normalised by N (bag-size invariant; duplicating the bag is a no-op on the
mean). Deterministic at inference (dropout off; P is a buffer).

Param count (input_dim=1280, hidden=128, r=3)
---------------------------------------------
    bottleneck Linear(1280,128)+b = 163,968   (trainable)
    head       Linear(128,1)+b    =     129   (trainable)
    projector  P [3,1280]         = buffer, FROZEN, not counted as trainable
    --------------------------------------------------------------
    total trainable               = 164,097   (< 197,250 baseline; <= ~197K)
The subspace projector adds ZERO trainable parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_blob(path: Path):
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _grade_subspace_basis(blob, input_dim: int, rank: int) -> Optional[torch.Tensor]:
    """Orthonormal rows P [r, input_dim] spanning the centered grade-mean subspace.

    M = [c_0..c_3] - mean_g c_g  (rank 3 by construction); P = top-r right
    singular vectors of M. Train-only (seed=2 prototypes), init-only.
    """
    if blob is None or "prototypes" not in blob:
        return None
    protos = blob["prototypes"]
    keys = sorted(protos.keys())
    C = torch.stack([protos[g].float().view(-1) for g in keys], dim=0)  # [G, D]
    if C.shape[1] != input_dim:
        return None
    M = C - C.mean(dim=0, keepdim=True)                # [G, D] centered grade means
    # Right singular vectors (rows of Vt) are the directions in feature space.
    _, S, Vt = torch.linalg.svd(M, full_matrices=False)  # Vt: [min(G,D), D]
    r = int(min(rank, Vt.shape[0]))
    # Keep only components with non-trivial singular value (drop the ~0 one).
    P = Vt[:r].contiguous()                            # [r, D], orthonormal rows
    return P


def _fibrosis_axis(blob, input_dim: int) -> Optional[torch.Tensor]:
    if blob is None or "axis" not in blob:
        return None
    v = blob["axis"].float().view(-1)
    if v.numel() != input_dim:
        return None
    return v / v.norm().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        subspace_rank: int = 3,        # r: grade-subspace dim (a94 ablation: project=False)
        use_subspace: bool = True,     # a93 main = True; a94 ablation = False (identity)
        warm_start: bool = True,       # init head along (in-subspace) train fibrosis axis
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,    # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        self.use_subspace = bool(use_subspace)
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here, byte-identical to a88/a92).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        blob = _load_blob(path)

        # Frozen rank-r grade-subspace projector P [r, D]. If the cache is
        # absent or use_subspace is off, P is None and the forward skips it.
        P = _grade_subspace_basis(blob, input_dim, subspace_rank) if self.use_subspace else None
        if P is not None:
            self.register_buffer("P", P)          # orthonormal rows, FROZEN
        else:
            self.P = None
            self.use_subspace = False

        # Warm-start the head along the IN-SUBSPACE fibrosis response (train-only).
        if warm_start:
            axis = _fibrosis_axis(blob, input_dim)
            if axis is not None:
                with torch.no_grad():
                    if self.use_subspace and self.P is not None:
                        axis = (axis @ self.P.t()) @ self.P  # restrict to span(P) (no-op math)
                        axis = axis / axis.norm().clamp(min=1e-8)
                    w1 = self.bottleneck[0].weight.detach()   # [hidden, input]
                    resp = w1 @ axis                          # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    self.head.weight.copy_(resp.view(1, -1) * 3.0)
                    self.head.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        x = features
        if self.use_subspace and self.P is not None:
            # Orthogonal projection onto the rank-r grade-subspace:
            #   x_proj = (x P^T) P   == x restricted to span(P), complement dropped.
            x = (x @ self.P.t()) @ self.P            # [N, D]

        h = self.bottleneck(x)                       # [N, hidden]
        z = h.mean(dim=0)                            # [hidden] DIFFUSE pool (plain mean)
        y = self.head(z).view(-1)                    # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    subspace_rank=3,       # a93 main = rank-3 grade subspace
    use_subspace=True,     # a93 main = project; a94 ablation = False (identity / full 1280-d)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
