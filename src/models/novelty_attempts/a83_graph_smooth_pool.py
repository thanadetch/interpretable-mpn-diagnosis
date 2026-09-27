"""a83 — Multi-step learnable feature-graph smoothing, then diffuse pool.

Angle (extends a59, the session-best at val 0.8128 = "feature-graph
smoothness"). Reticulin fibrosis is graded as the OVERALL / DIFFUSE density of
a *connected* fibre meshwork across the whole marrow, NOT a property of a few
patches. a59 exploited this only implicitly (a one-shot consensus-consistency
softmax weight). a83 makes the contiguity prior EXPLICIT and MULTI-STEP: build
a patch-similarity graph in BOTTLENECK space and run K=2 steps of learnable
graph smoothing (a graph low-pass / label-propagation step), so that fibrosis
evidence that is CONTIGUOUS / mutually-consistent across many patches is
reinforced, while ISOLATED outlier patches (lone bone fragments, edge/stain
artefacts that do not agree with their neighbours) are blended away. Then we
DIFFUSE-POOL (plain mean of the smoothed representation) and read out a grade
with a linear head. This is the natural extension a59 explicitly asked for:
replace the single consensus step with iterated neighbourhood averaging.

Mechanism:
    h_i  = Dropout(ReLU(Linear(1280->128) f_i))         # baseline encoder
    A    = rownorm( relu(cos(h_i, h_j)) )                # [N,N] affinity, row-sum=1
                                                          # (parameter-free, data-driven)
    for k in 1..K:                                        # K learnable smoothing steps
        h <- (1 - a) * h + a * (A @ h)                    # a in [0,1] (learnable)
    z    = mean_i h_i                                     # DIFFUSE pool (whole-bag density)
    y    = Linear(128 -> 1)(z)                            # raw grade logit

Key design choices (each motivated by a documented diagnostic finding):
  - Affinity A in BOTTLENECK space, cosine-based => MAGNITUDE-INVARIANT. We do
    NOT weight by feature-norm ||h|| (Spearman ~0 with grade => nuisance, tracks
    tissue-vs-background not fibrosis). relu(cos) keeps only non-negative
    (agreeing) edges so anti-correlated outliers are not propagated.
  - Row-normalisation makes A@h a convex neighbourhood average => the smoothing
    step is a contraction that CANNOT explode (bounded by max |h|), so K steps
    stay finite even at real Virchow2 feature scale (~6). No NaN risk.
  - The mixing coefficient `a` is the ONLY new learnable scalar (per the K=0
    ablation we flip it off structurally). a=0 recovers the plain diffuse pool;
    a>0 trades raw evidence for neighbourhood-consistent evidence. Learnable so
    the model can choose how much contiguity to trust.
  - Warm-start: the linear head's weight is initialised along the train-only
    fibrosis axis (data/prototypes_virchow2_reti_train_seed2.pt['axis'],
    1280-d, seed=2) projected onto the bottleneck's input weights, i.e. we
    point the readout at the fibrosis direction from step 0. This is a pure
    INITIALISATION (then fully learned); it injects no test information and uses
    ONLY the train-derived axis.

Why this is a genuinely FRESH angle (checked against the refuted list):
  - NOT rank/norm-salience, NOT top-k, NOT per-patch severity: there is no
    per-patch scorer, no selection, no argmax — every patch enters the diffuse
    mean equally AFTER smoothing.
  - NOT plain coverage/moments/quantiles/pairwise/dist-match: the readout is a
    single linear map of ONE diffuse-pooled vector; no threshold/sigmoid
    fraction, no concatenated moments/quantiles, no pairwise statistic, no
    distribution matching.
  - NOT consensus (a59/a62): a59 uses a SINGLE cosine-to-centroid softmax
    WEIGHT then weighted-mean of severities; a83 uses ITERATED row-normalised
    neighbourhood propagation A@h that REWRITES each patch's representation
    using its agreeing neighbours (a graph low-pass), then an UNWEIGHTED diffuse
    mean. Different object (a transformed h, not a weight) and multi-step.
  - NOT PMA / learned-query attention: A is parameter-free (data-driven cosine),
    not a learned seed/query; pooling is the plain mean.
  - NOT a gate on the output: the head is linear in the pooled vector; nothing
    multiplies the scalar.

Permutation-invariance: A is built from pairwise cosines and row-normalised, so
permuting patches permutes rows/cols of A consistently; A@h is permutation-
EQUIVARIANT, and the final mean over patches is permutation-INVARIANT.
Bag-size-invariance: cosine affinity and row-normalisation are scale-free in N;
the final mean normalises by N. Duplicating the bag leaves the smoothed mean
unchanged up to numerical noise. Deterministic at inference (dropout off).

Ablation companion: a84 = K=0 (no smoothing) == plain diffuse mean-pool +
linear head. a83 (K=2) vs a84 (K=0) isolates EXACTLY the active ingredient =
"does multi-step graph smoothing of the contiguous fibre signal beat the plain
diffuse pool?". The bottleneck encoder and head are byte-identical between them.

Kill criterion: abandon if a83 val_qwk < 0.78 at seed=2 AND a83 <= a84
(smoothing adds nothing over the diffuse pool). DoD = multi-seed audit, not
seed=2 alone.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    head       Linear(128,1)+b     =    129
    mix_logit  scalar (-> a)       =      1
    --------------------------------------------
    total                          = 164,098   (< 197,250 baseline)
The affinity / smoothing steps add ZERO learnable parameters except `a`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis (1280-d) from train-only prototypes (seed=2).

    Returns None if the cache is absent so the model still constructs (head
    then uses its default init). Uses ONLY the train-derived axis — no test
    information, no runtime features.
    """
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
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
        smooth_steps: int = 2,           # K: graph-smoothing steps (a84 ablation = 0)
        mix_init: float = 0.5,           # initial mixing coeff a in [0,1] (learnable)
        warm_start: bool = True,         # init head along train fibrosis axis
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,      # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert smooth_steps >= 0, "smooth_steps (K) must be >= 0."
        self.smooth_steps = int(smooth_steps)
        self.clamp_output = clamp_output

        # Baseline-identical encoder (this is where the capacity lives).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Linear readout from the diffuse-pooled (smoothed) representation.
        self.head = nn.Linear(hidden_dim, num_classes)

        # Learnable mixing coefficient a in [0,1] via a sigmoid(logit).
        # logit chosen so sigmoid(logit) == mix_init at start.
        m = float(min(max(mix_init, 1e-4), 1.0 - 1e-4))
        self.mix_logit = nn.Parameter(torch.tensor(torch.logit(torch.tensor(m)).item()))

        # Warm-start the head toward the fibrosis axis (train-only, seed=2).
        # The bottleneck input weight W1 [hidden, input] maps raw features into
        # the bottleneck; W1 @ axis gives the bottleneck-space response to the
        # fibrosis direction. Pointing the head at that response means y starts
        # as a linear readout of the diffuse fibrosis density.
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_fibrosis_axis(path, input_dim)
            if axis is not None:
                with torch.no_grad():
                    w1 = self.bottleneck[0].weight.detach()        # [hidden, input]
                    resp = w1 @ axis                               # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    # Scale so the initial readout spans the grade range loosely.
                    self.head.weight.copy_(resp.view(1, -1) * 3.0)
                    self.head.bias.fill_(1.5)                      # mid-grade start

    @staticmethod
    def _affinity(h: torch.Tensor) -> torch.Tensor:
        """Row-normalised non-negative cosine affinity A [N,N] (data-driven).

        relu(cos) keeps only agreeing (non-negative) edges; each row sums to 1
        so A@h is a convex neighbourhood average (a contraction => finite).
        """
        hn = F.normalize(h, dim=1, eps=1e-8)        # [N, hidden]
        sim = hn @ hn.t()                           # [N, N] cosine in [-1, 1]
        sim = F.relu(sim)                           # drop anti-correlated edges
        row = sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return sim / row                            # row-stochastic

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        h = self.bottleneck(features)               # [N, hidden]

        n = h.size(0)
        if self.smooth_steps > 0 and n > 1:
            a = torch.sigmoid(self.mix_logit)       # mixing coeff in (0, 1)
            A = self._affinity(h)                   # [N, N] row-stochastic
            for _ in range(self.smooth_steps):
                # Convex update: stays within the convex hull of {h_i} => bounded.
                h = (1.0 - a) * h + a * (A @ h)

        z = h.mean(dim=0)                           # [hidden] DIFFUSE pool
        y = self.head(z).view(-1)                   # [1] RAW grade logit
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
    smooth_steps=2,        # a83 main = K=2 learnable graph smoothing
    mix_init=0.5,
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
