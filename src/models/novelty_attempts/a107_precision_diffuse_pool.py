"""a107 — Precision-weighted (inverse-within-bag-dispersion) DIFFUSE pool.

LENS = WILDCARD-ESCAPE. A parameter-free, closed-form, deterministic, *fold-
invariant* reshaping of the diffuse mean-pool that the SAME baseline head reads
at <= baseline degrees-of-freedom. No new learnable parameter, no seed=2-derived
direction, no attention/selection/salience. The ONLY thing that changes vs the
plain diffuse mean-pool baseline (= a108) is that each *bottleneck dimension* of
the pooled vector is reweighted by the INVERSE of its WITHIN-BAG dispersion
(a James-Stein / precision shrink), computed from THIS bag's own second moment.

Why this is a genuine grading prior (not a tied reshuffle)
----------------------------------------------------------
Reticulin grade is the OVERALL / DIFFUSE fibre density of the marrow. A
bottleneck dimension that fires *consistently across all patches* of a bag (LOW
within-bag dispersion) is, by definition, encoding a bag-WIDE / diffuse property
-> exactly the grading target. A dimension with HIGH within-bag dispersion fires
on some patches and not others -> it is encoding patch-IDIOSYNCRATIC content
(a lone bone fragment, a vessel, a stain/edge artefact) -> nuisance for a DIFFUSE
grade. The plain diffuse mean weights every dimension by its raw mean activation
regardless of whether that activation is diffuse or idiosyncratic. a107 instead
upweights the consensus (diffuse) dimensions and downweights the idiosyncratic
ones, per bag, in closed form. This is the precision (inverse-variance / Fisher)
weighting of the pooled estimate: the consensus directions are the ones the bag
estimates with low variance, so they get more weight.

Exact forward math (single bag features f in R^{N x 1280})
----------------------------------------------------------
    h_i   = Dropout(ReLU(W1 f_i + b1))            # [N,128] baseline encoder
    z     = mean_i h_i                            # [128]   plain diffuse mean
    if N >= 2:
        v     = var_i h_i  (population, unbiased=False)  # [128] within-bag dispersion
        vbar  = mean_d v_d                                # scalar (bag dispersion scale)
        v_s   = (1 - lam) * v + lam * vbar                # shrink dispersion toward vbar
        w     = vbar / (v_s + eps)                        # [128] precision weight, in (0, 1/(1-lam)]
        w     = w / mean_d w_d                            # MEAN-1 normalise => preserves overall scale of z
        z_w   = z * w                                     # [128] precision-weighted diffuse mean
    else:
        z_w   = z                                         # N<2: no dispersion => plain mean
    y     = W2 z_w + b2                            # [1] RAW grade logit

`lam in [0,1]` is the shrinkage strength toward the bag's mean dispersion `vbar`.
At lam=1, v_s == vbar for every dimension, so w == 1 uniformly and z_w == z
EXACTLY (the plain diffuse mean-pool). At lam<1, high-dispersion dims are
down-weighted relative to low-dispersion (diffuse) dims. The mean-1 renorm
(`w / mean(w)`) keeps the overall magnitude of z_w equal to that of z, so the
head reads a vector at the SAME scale as the baseline — only the per-dimension
BALANCE changes. lam is a FIXED python float (a structural hyperparameter), NOT
an nn.Parameter: it adds ZERO degrees of freedom and CANNOT drift to fit a fold.

How a107 ESCAPES the UNDERFIT trap
----------------------------------
The full baseline encoder W1: 1280->128 (163,968 params) is retained verbatim,
so the model keeps its complete expressivity (unlike the frozen-axis 2-param
a101, the 6-param scatter pool, the no-bottleneck quantiles a70, or the rank-3
projector a93 that throttles the input to 3 dims). The reweight is a 128-d,
data-rich closed-form statistic of the bag, not a 1.3K-param coverage scalar or
a 2-DOF affine. There is no low-dimensional information bottleneck imposed on the
readout: the head still sees a full 128-d pooled vector. So a107 has at least the
baseline's fitting capacity -> it cannot underfit the way the low-DOF readouts
did (which is why those landed at val 0.0-0.77).

How a107 ESCAPES the VAL-OVERFIT trap
-------------------------------------
The richer mechanisms that lifted seed=2 val and tanked test (subspace a93 0.841,
spectral a99 0.840, heavy-reg a91 0.839; also every attention/PMA/learned-query
variant) all gave the head EXTRA learnable degrees of freedom (a learned
projector, a learned spectral filter, a wider/structured readout, a learned
gate) which the small/hard seed=2 val fold could be driven to fit. a107 adds
*zero* learnable degrees of freedom over the baseline: the head is the SAME
Linear(128,1) (129 params), the encoder is byte-identical, and the reweight `w`
is a deterministic, parameter-free function of the bag with NO knob the optimiser
can turn toward the seed=2 fold. There is literally nothing fold-specific to
overfit. The reweight is computed by the IDENTICAL closed-form formula on every
bag of every fold (it is not derived from the seed=2 axis/prototypes, so it does
not wash out as the favorable-leaky warm-start did in a103). Any gain therefore
has to come from a fold-INVARIANT structural prior (suppress idiosyncratic
dims, keep diffuse dims), which is exactly the kind of mechanism that can move
the PAIRED cross-fold Δ rather than just the seed=2 val number.

Why it is NOT a known tied relabel
----------------------------------
- NOT L2-norm / DE10 / a81: those rescale each PER-PATCH vector by its norm
  BEFORE the mean. a107 never touches per-patch norms; it reweights POOLED
  DIMENSIONS by their inverse within-bag dispersion AFTER the mean. Different
  object, different axis.
- NOT mean+std concat / moments (DE07, a64): those CONCATENATE dispersion to the
  mean and feed a WIDER head (extra head DOF -> val-overfit). a107 FOLDS the
  dispersion back into the SAME 128-d vector and feeds the SAME 129-param head
  (zero extra DOF).
- NOT subspace/spectral (a93/a99): those use a FROZEN seed=2-derived projector /
  filter (leaky, fold-specific). a107's weight is the bag's OWN second moment,
  recomputed per bag, identical formula across folds, no external direction.
- NOT attention/top-k/severity/consensus/graph-smooth/rank/coverage: there is no
  per-patch scorer, no selection, no softmax weight, no neighbourhood step. Every
  patch enters the mean equally; only the bag-level dimension balance is reshaped.
- NOT heavy-reg (a91): no train-time noise, no dropout change; the reweight is ON
  at inference and is a representation prior, not stochastic regularisation.

Permutation- & bag-size-invariance
-----------------------------------
Both the mean and the population variance over patches are symmetric functions of
the rows, so permuting patches leaves z, v (hence w, z_w, y) unchanged. Both are
normalised by N (mean) / are intensive statistics (per-dim variance), so
duplicating the whole bag leaves z and v EXACTLY unchanged -> bag-size invariant.
Deterministic at inference: dropout is off under model.eval(); the reweight is a
fixed closed-form function with no sampling. Two eval passes are bit-identical.
N<2 (no within-bag dispersion) falls back to the plain mean (no NaN).

Warm-start (init-only, train-only seed=2; identical between a107/a108)
----------------------------------------------------------------------
The head weight is initialised along the bottleneck response to the train-only
fibrosis axis (a83 idiom), so the readout points at the diffuse fibrosis
direction from step 0. Pure initialisation, fully learned after; byte-identical
in main and ablation so it does NOT confound the active-ingredient isolation.
(Per the a103 finding, warm-start helps only seed=2 and washes out paired — so
it is NOT relied on for the paired gain; it is here only to match the baseline's
starting point and keep the ablation clean.)

Ablation companion (a108): lam=1.0 -> w == 1 -> z_w == z EXACTLY = the plain
diffuse mean-pool baseline, with byte-identical encoder, head and warm-start.
a107 (lam=0.5) vs a108 (lam=1.0) isolates EXACTLY the active ingredient:
"does inverse-within-bag-dispersion (precision) reweighting of the diffuse pool
beat the plain diffuse mean?".

Kill criterion: abandon if a107 does NOT beat a108 on PAIRED cross-fold Δ
(both val & test, majority of folds {0,1,2,3,42}); a single seed=2 val lift is
explicitly NOT a win (a84/a96 cleared seed=2 yet lost paired).

Param count (input_dim=1280, hidden=128)
----------------------------------------
    bottleneck Linear(1280,128)+b = 163,968   (trainable)
    head       Linear(128,1)+b    =     129   (trainable)
    -----------------------------------------------------
    total trainable               = 164,097   (< 197,250 baseline; <= ~197K)
The precision reweight adds ZERO learnable parameters (lam is a fixed float).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROTOTYPE_PATH = _REPO_ROOT / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _load_fibrosis_axis(path: Path, input_dim: int) -> Optional[torch.Tensor]:
    """Unit fibrosis axis (1280-d) from train-only prototypes (seed=2).

    Returns None if the cache is absent so the model still constructs. Uses ONLY
    the train-derived axis for INITIALISATION; injects no test information and no
    runtime features.
    """
    if not path.is_file():
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        v = blob["axis"].float().view(-1)
    except Exception:
        return None
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
        shrink_lambda: float = 0.5,   # lam: shrink dispersion toward vbar. a108 ablation = 1.0
        warm_start: bool = True,      # init head along train fibrosis axis (a83 idiom)
        prototype_path: Optional[str] = None,
        eps: float = 1e-6,
        clamp_output: bool = False,   # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert 0.0 <= shrink_lambda <= 1.0, "shrink_lambda must be in [0, 1]."
        # lam is a FIXED float (a structural hyperparameter) — NOT an nn.Parameter.
        # It therefore adds zero degrees of freedom and cannot drift to fit a fold.
        self.shrink_lambda = float(shrink_lambda)
        self.eps = float(eps)
        self.clamp_output = clamp_output

        # Baseline-identical diffuse encoder (this is where the capacity lives).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Linear readout from the (precision-weighted) diffuse-pooled vector.
        self.head = nn.Linear(hidden_dim, num_classes)

        # Warm-start the head toward the in-bottleneck fibrosis response (train-only).
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_fibrosis_axis(path, input_dim)
            if axis is not None:
                with torch.no_grad():
                    w1 = self.bottleneck[0].weight.detach()   # [hidden, input]
                    resp = w1 @ axis                          # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    self.head.weight.copy_(resp.view(1, -1) * 3.0)
                    self.head.bias.fill_(1.5)                 # mid-grade start

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        h = self.bottleneck(features)                # [N, hidden]
        z = h.mean(dim=0)                            # [hidden] plain diffuse mean

        n = h.size(0)
        if n >= 2:
            # Within-bag per-dimension dispersion (population variance over patches).
            # Permutation-symmetric and intensive (bag-size invariant).
            v = h.var(dim=0, unbiased=False)         # [hidden]
            vbar = v.mean().clamp(min=self.eps)      # scalar bag dispersion scale
            # James-Stein-style shrink of the dispersion toward vbar. At lam=1 the
            # shrunk dispersion is vbar for every dim => the weight collapses to 1.
            v_shrunk = (1.0 - self.shrink_lambda) * v + self.shrink_lambda * vbar
            # Precision weight: low-dispersion (diffuse/consensus) dims get more
            # weight; high-dispersion (patch-idiosyncratic) dims get less.
            w = vbar / (v_shrunk + self.eps)         # [hidden]
            # Mean-1 normalisation: preserve the overall scale of z (only the
            # per-dimension BALANCE changes), so the head reads a baseline-scale
            # vector and gains no extra effective magnitude DOF.
            w = w / w.mean().clamp(min=self.eps)     # [hidden], mean == 1
            z = z * w                                # precision-weighted diffuse mean

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
    shrink_lambda=0.5,     # a107 main = precision reweight ON. a108 ablation = 1.0 (OFF -> plain mean)
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    eps=1e-6,
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
