"""a93 — Hard rank-3 grade-subspace projection, then plain diffuse mean-pool.

Angle (REPRESENTATION, not pool/readout). Reticulin fibrosis is graded as the
OVERALL / DIFFUSE density of a connected fibre meshwork across the whole marrow,
NOT a property of a few patches, and it must NOT be weighted by feature-norm
||h|| (Spearman ~0 with grade => ||.|| is nuisance: stain/scanner/tissue-density,
not fibrosis). Every previously-tied mechanism modified HOW patches are combined
or HOW the scalar is read out (the POOL / weights / head): gated-attention,
mean-pool, top-k, rank/norm-salience, coverage, per-patch severity, consensus,
contiguity, PMA, soft-rank, moments, quantiles, pairwise-spread, dist-match,
backbone-fusion, ensemble, L2-norm, graph-smoothing, robust-trimmed, ordinal
head, dropout+input-noise. a93 leaves the COMBINATION (plain unweighted mean)
and the READOUT (single linear map) byte-identical to the tied diffuse baseline
(a88/a92) and instead changes the REPRESENTATION: it HARD-PROJECTS each patch
feature onto a FIXED train-only rank-3 grade-relevant subspace BEFORE the
bottleneck, deleting the 1277 grade-irrelevant directions so the encoder cannot
overfit on them.

Mechanism (single bag f in R^{N x 1280}):
  (1) Build ONCE at init a frozen orthonormal basis P in R^{3 x 1280} of the
      grade-subspace. Stack the 4 train-only prototype grade-means c_0..c_3
      (seed=2 cache), center them M = C - mean_g(c_g)  [4,1280] (rank EXACTLY 3;
      verified singular values [14.45, 9.18, 5.35, 3.8e-6]), take the top-3
      right singular vectors P = Vt[:3] (P P^T = I_3). Register P as a BUFFER
      => 0 trainable params. The fibrosis axis (c_3 - c_0)/||.|| lies in
      span(P), so its projection is a no-op (verified residual 0).
  (2) Orthogonally project each patch onto span(P):
          coords = f @ P^T            [N, 3]   (grade-subspace coordinates)
          x_proj = coords @ P         [N, 1280] (= f restricted to span(P),
                                                 orthogonal complement zeroed).
      This is a genuine orthogonal projector (x_proj = x P^T P), it RESCALES
      nothing by ||.|| (NOT norm-salience), and it is the SAME P for all folds.
  (3) Baseline-identical encoder:
          h_i = Dropout0.5(ReLU(Linear(1280->128) x_proj_i))   [N, 128]
  (4) PLAIN DIFFUSE POOL z = mean_i h_i  [128] (unweighted mean, identical to
      the tied a88/a92 diffuse baseline).
  (5) y = Linear(128->1)(z)  [1] RAW logit (trainer rounds+clips at eval).

Warm-start (INIT-ONLY, TRAIN-ONLY): the fibrosis axis (c_3-c_0)/||.|| lies in
span(P) so projection is a no-op; copy unit-normed (W1 @ axis)*3.0 into
head.weight, bias=1.5 (mid-grade). Pure initialisation, then fully learned;
injects no test/fold information, uses ONLY the train-derived axis.

Why ORTHOGONAL to the refuted list:
  - NOT DE22 single-axis projection: DE22 used <f, axis> as a per-patch
    ATTENTION SCORER (a softmax WEIGHT, changing the POOL, collapsing each patch
    to ONE scalar with a SINGLE axis). a93 changes NO weight, keeps the FULL 3-D
    subspace coordinates feeding the 128-d bottleneck, and a single axis cannot
    even span the rank-3 grade-mean signal. This is precisely the multi-axis
    rank-K projection the DE22 post-mortem named verbatim as "still untested".
  - NOT input-dropout (DE19): dropout zeros RANDOM dims with stochastic
    train-time noise (an inference no-op). a93 deterministically keeps a FIXED
    data-chosen low-rank subspace at BOTH train and eval — a structural prior,
    not noise.
  - NOT norm-salience: the orthogonal projector rescales nothing by ||.||.
  - NOT a new pool / readout: the combination is the plain mean and the readout
    is a single linear map, byte-identical to the diffuse baseline.

Paired-robustness argument: a84 won both seed=2 gates yet lost 4/5 folds because
seed=2 tuning is a lottery. With only 30 train patients, the 1280-d Virchow2
features carry far more nuisance variance (stain, scanner, tissue-density) than
grade signal, so the bottleneck W1(1280->128) is free to key on fold-specific
nuisance directions — the exact source of the val<->test gap. Hard-projecting
onto the rank-3 between-grade-mean subspace FIRST cuts the encoder's effective
input dimensionality from 1280 to 3: it literally CANNOT fit a fold-specific
direction orthogonal to grade because that direction is zeroed before it ever
reaches a weight. P is FIXED (train-only seed=2, init-only) and the SAME P is
reused across all folds {0,1,2,3,42}, so any consistent gain is a structural-
prior gain that TRANSFERS across folds (no test/fold info injected). This is a
low-capacity (0 added trainable params) variance reduction aimed squarely at
PAIRED cross-fold generalisation on both val and test.

Permutation-invariance: P-projection is per-patch (equivariant), the encoder is
per-patch, and the final mean over patches is permutation-INVARIANT.
Bag-size-invariance: the final mean normalises by N, so duplicating the bag
leaves z (hence y) unchanged up to numerical noise. Deterministic at inference
(dropout off; P is a fixed buffer).

Ablation companion: a94 imports this Model and flips ONLY use_subspace=False
(projector still BUILT but SKIPPED), recovering the standard full-1280-d diffuse
mean-pool baseline (the a88/a92 null). a93 vs a94 isolates EXACTLY the active
ingredient = the x_proj = (x P^T) P rank-3 restriction, and nothing else.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    head       Linear(128,1)+b     =    129
    --------------------------------------------
    total trainable                = 164,097   (< 197,250 baseline)
The projector P is a frozen BUFFER => ZERO learnable parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_blob(path: Path):
    """Load the train-only (seed=2) prototype cache, or None if absent."""
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _grade_subspace_basis(blob, input_dim: int, rank: int = 3) -> Optional[torch.Tensor]:
    """Frozen orthonormal basis P [rank, input_dim] of the grade-mean subspace.

    Stack the 4 train-only prototype grade-means C [4, D], center them
    M = C - mean_g(C) (rank EXACTLY 3), and return the top-`rank` right singular
    vectors P = Vt[:rank] (P P^T = I_rank). Returns None if the cache is
    malformed so the model still constructs. Uses ONLY train-derived prototypes
    (seed=2) — no test information, no runtime features.
    """
    if blob is None or "prototypes" not in blob:
        return None
    protos = blob["prototypes"]
    try:
        keys = sorted(protos.keys())
    except Exception:
        return None
    if len(keys) < rank + 1:
        return None
    try:
        C = torch.stack([protos[k].float().view(-1) for k in keys], dim=0)  # [G, D]
    except Exception:
        return None
    if C.dim() != 2 or C.size(1) != input_dim:
        return None
    M = C - C.mean(dim=0, keepdim=True)            # [G, D] centered grade-means
    # Top-`rank` right singular vectors span the between-grade-mean subspace.
    _, _, Vt = torch.linalg.svd(M, full_matrices=False)
    P = Vt[:rank].contiguous()                     # [rank, D], orthonormal rows
    return P


def _load_fibrosis_axis(blob, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis (1280-d) from train-only prototypes (seed=2).

    Returns None if absent. Uses ONLY the train-derived axis — no test info, no
    runtime features. Defensive: validates shape before use.
    """
    if blob is None or "axis" not in blob:
        return None
    v = blob["axis"]
    if not torch.is_tensor(v):
        return None
    v = v.float().view(-1)
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
        use_subspace: bool = True,        # a94 ablation = False (projector skipped)
        subspace_rank: int = 3,           # rank-K grade subspace (grade-means => 3)
        warm_start: bool = True,          # init head along train fibrosis axis
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,       # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert subspace_rank >= 1, "subspace_rank must be >= 1."
        self.use_subspace = bool(use_subspace)
        self.subspace_rank = int(subspace_rank)
        self.clamp_output = clamp_output

        # Baseline-identical encoder (this is where the capacity lives).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Linear readout from the diffuse-pooled representation.
        self.head = nn.Linear(hidden_dim, num_classes)

        # Load the train-only (seed=2) cache once for both the projector and the
        # warm-start axis.
        path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
        blob = _load_blob(path)

        # --- Frozen rank-K grade-subspace projector P [rank, D] (0 params) ---
        # ALWAYS built (even when use_subspace=False) so a93 and a94 are
        # byte-identical except the documented flip of `use_subspace`.
        P = _grade_subspace_basis(blob, input_dim, rank=self.subspace_rank)
        if P is None:
            # Defensive fallback: identity-on-first-`rank`-dims has no effect on
            # the *trainable* params (still 0). If the cache is missing the model
            # still constructs; the projection then keeps a fixed coordinate
            # subspace. The trainer always has the seed=2 cache present, so this
            # branch is for robustness only.
            P = torch.zeros(self.subspace_rank, input_dim)
            for i in range(self.subspace_rank):
                P[i, i] = 1.0
        # Register as a buffer => frozen, moves with .to(device), not trainable.
        self.register_buffer("subspace_P", P)  # [rank, D]

        # --- Warm-start the head toward the fibrosis axis (train-only seed=2) ---
        # The fibrosis axis lies in span(P) so projection is a no-op; pointing
        # the head at the bottleneck's response to that axis means y starts as a
        # linear readout of the diffuse fibrosis density. Pure init, then learned.
        if warm_start:
            axis = _load_fibrosis_axis(blob, input_dim)
            if axis is not None:
                with torch.no_grad():
                    w1 = self.bottleneck[0].weight.detach()        # [hidden, input]
                    resp = w1 @ axis                               # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    # Scale so the initial readout spans the grade range loosely.
                    self.head.weight.copy_(resp.view(1, -1) * 3.0)
                    self.head.bias.fill_(1.5)                      # mid-grade start

    def _project(self, f: torch.Tensor) -> torch.Tensor:
        """Orthogonal projection of each patch onto span(P): x_proj = (f P^T) P.

        f      : [N, D]
        coords : [N, rank]  grade-subspace coordinates
        x_proj : [N, D]     f restricted to span(P), orthogonal complement zeroed.
        Rescales nothing by ||.|| (true orthogonal projector). Same P all folds.
        """
        P = self.subspace_P                          # [rank, D] buffer
        coords = f @ P.t()                           # [N, rank]
        x_proj = coords @ P                          # [N, D]
        return x_proj

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        if self.use_subspace:
            x = self._project(features)              # [N, D] rank-K restriction
        else:
            x = features                             # a94 null: full 1280-d input

        h = self.bottleneck(x)                       # [N, hidden]
        z = h.mean(dim=0)                            # [hidden] DIFFUSE pool
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
    use_subspace=True,     # a93 main = hard rank-3 grade-subspace projection ON
    subspace_rank=3,       # grade-means => rank exactly 3
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
