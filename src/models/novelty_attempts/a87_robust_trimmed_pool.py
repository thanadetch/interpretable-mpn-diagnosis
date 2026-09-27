"""a87 — Outlier-robust (soft-trimmed) DIFFUSE feature pool.

WHY (robustness, not seed=2 signal). Every result this session says the
SIGNAL is already there (86 aggregators + 3 backbones + fusion all tie the
baseline at paired cross-fold Delta ~ 0); the bottleneck is VARIANCE /
generalisation — the val<->test gap, not val on one seed. a84 even cleared
both seed=2 gates yet was a seed=2 lottery (lost 4/5 folds paired) because it
was a *plain* diffuse mean over the bottleneck features, and the plain mean is
NOT robust: a handful of patches that do NOT belong to the diffuse tissue
manifold (lone bone trabeculae, fat, stain/edge/section artefacts) sit far from
the bag's feature centroid and drag the pooled vector by an amount that varies
fold-to-fold with how many such junk patches each ROI happens to contain. That
per-bag composition noise is exactly the kind of nuisance that inflates the
val<->test gap.

ANGLE. Reticulin grade = the OVERALL / DIFFUSE reticulin density of the marrow
tissue, not a property of a few patches and NOT magnitude (||h||) salience. So
we want a DIFFUSE pool (every tissue patch counts), but a ROBUST one: estimate
the *typical* tissue representation and down-weight feature-space outliers, so
the pooled vector tracks the bulk marrow manifold rather than the arithmetic
mean which a few junk patches can move.

MECHANISM (deterministic, no sampling, no selection):
    h_i = Dropout(ReLU(Linear(1280->128) f_i))      # baseline encoder
    # Robust feature centroid via a FIXED number of IRLS steps toward the
    # geometric median (Weiszfeld), initialised at the ordinary centroid:
    m_0 = mean_i h_i
    repeat T times:
        d_i      = || h_i - m_t ||                    # distance to current centroid
        u_i      = 1 / (1 + (d_i / r)^2)              # Cauchy/Welsch robust weight
        m_{t+1}  = sum_i u_i h_i / sum_i u_i          # reweighted centroid
    # Final SOFT-TRIMMED diffuse pool: weight each patch by closeness to the
    # robust centroid m_T, so feature-space outliers are down-weighted:
    d_i  = || h_i - m_T ||
    w_i  = softmax_i( - d_i^2 / s )                   # s>0 learnable (softplus)
    z    = sum_i w_i h_i                              # ROBUST diffuse pool
    y    = Linear(128 -> 1)(z)                        # RAW grade logit

`r` (Weiszfeld scale) and `s` (trimming temperature) are the ONLY new learnable
scalars (both softplus-positive). As s -> inf every w_i -> 1/N and the pool
becomes the PLAIN MEAN (a84) — so the mean is the strict no-robustness limit
and the a88 ablation flips to exactly that. As s shrinks, far-from-centre
patches are progressively trimmed. The IRLS step adds ZERO learnable matrices.

WHY OUTLIER-ROBUST DIFFUSE POOLING SHOULD GENERALISE BETTER THAN THE MEAN.
The trimmed mean is a classical M-estimator of LOCATION whose breakdown point
is higher than the arithmetic mean's (=0): a minority of arbitrarily-placed
junk patches cannot move it much. Because the val and test ROIs differ mainly
in nuisance composition (how much bone/fat/artefact is captured), a pool that
is by-construction insensitive to that minority should have a SMALLER
val<->test gap, i.e. lower per-fold variance — the win bar here.

DISTINCTNESS (checked against the refuted list + a58/a59/a62):
  - NOT a58: a58 takes a robust soft-median of per-patch SEVERITY SCALARS g_i
    (a 1-D location of scores). a87 robustly locates the FEATURE CENTROID in
    128-D and pools the FEATURE VECTORS h_i — the robustness acts in feature
    space on the representation, there is no per-patch severity scorer at all.
  - NOT a59/a62: those weight by cosine AGREEMENT with the (ordinary) centroid
    then take a weighted-mean of severities. a87 weights by EUCLIDEAN DISTANCE
    to a ROBUST (Weiszfeld geometric-median) centroid and pools features; the
    reference is the outlier-resistant median, not the (outlier-sensitive)
    mean, and the readout is a single linear map of one pooled vector.
  - NOT norm/rank salience, NOT top-k, NOT per-patch severity/consensus, NOT
    coverage/moments/quantiles/pairwise/dist-match, NOT PMA/learned-query, NOT
    graph-smooth (a83), NOT L2-norm, NOT ensemble: there is no per-patch score,
    no selection/argmax, no threshold/fraction, no concatenated moments, no
    pairwise statistic, no learned query/seed, no graph propagation A@h, no
    feature-norm weighting, no model averaging. Every patch enters one diffuse
    weighted mean; the weight is a residual-distance trimming, not magnitude.
  - DETERMINISTIC at inference: dropout is the only stochastic op and it is off
    in eval(); IRLS is a fixed-iteration closed-form recurrence (no sampling,
    no MC). The forward uses ONLY `features`.

PERMUTATION- & BAG-SIZE-INVARIANCE. m_0, every IRLS update, the softmax over i
and the final pool are all symmetric sums over patches => permuting patches
leaves z (hence y) unchanged. Distances/weights depend only on each h_i and the
shared centroid; duplicating the bag leaves m_T and the normalised weights
unchanged (every step is a normalised weighted average), so the pool is
bag-size-invariant. detach() on m_t inside the weight (see code) keeps the
recurrence a fixed point estimate and the gradient well-behaved; it does not
break either invariance.

WARM-START. Same idiom as a83/a84: initialise the linear head along the
train-only fibrosis axis (seed=2 prototypes) projected through the bottleneck's
input weights, so the readout points at the diffuse fibrosis direction from
step 0. Pure initialisation (then fully learned); injects NO test information,
uses ONLY the train-derived axis; absent cache => default init, still builds.

ABLATION (a88) = robust=False => uniform weights == PLAIN MEAN-POOL (a84). a87
(robust trimmed) vs a88 (mean) isolates EXACTLY: "does outlier-robust diffuse
pooling generalise better (smaller per-fold variance) than the plain mean?"
The bottleneck encoder, head and warm-start are byte-identical between them.

Param count (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    head       Linear(128,1)+b     =    129
    log_r scalar (-> r)            =      1
    log_s scalar (-> s)            =      1
    --------------------------------------------
    total                          = 164,099   (< 197,250 baseline)
The IRLS / trimming steps add ZERO learnable matrices.
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
        robust: bool = True,          # a88 ablation = False => plain mean-pool
        irls_steps: int = 3,          # T: Weiszfeld IRLS steps toward geo-median
        scale_init: float = 1.0,      # init Weiszfeld scale r (softplus-positive)
        trim_init: float = 4.0,       # init trimming temperature s (softplus-positive)
        warm_start: bool = True,      # init head along train fibrosis axis
        prototype_path: Optional[str] = None,
        clamp_output: bool = False,   # RAW logits; trainer rounds/clips at eval
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert irls_steps >= 0, "irls_steps (T) must be >= 0."
        self.robust = bool(robust)
        self.irls_steps = int(irls_steps)
        self.clamp_output = clamp_output

        # Baseline-identical encoder (this is where the capacity lives).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Linear readout from the (robust) diffuse-pooled representation.
        self.head = nn.Linear(hidden_dim, num_classes)

        # Two positive scalars via softplus(log_*): the Weiszfeld scale r and
        # the trimming temperature s. softplus keeps them strictly positive and
        # differentiable. inverse-softplus init so softplus(log_*) == *_init.
        self.log_r = nn.Parameter(torch.tensor(_inv_softplus(scale_init)))
        self.log_s = nn.Parameter(torch.tensor(_inv_softplus(trim_init)))

        # Warm-start the head toward the fibrosis axis (train-only, seed=2).
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_fibrosis_axis(path, input_dim)
            if axis is not None:
                with torch.no_grad():
                    w1 = self.bottleneck[0].weight.detach()        # [hidden, input]
                    resp = w1 @ axis                               # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    self.head.weight.copy_(resp.view(1, -1) * 3.0)
                    self.head.bias.fill_(1.5)                      # mid-grade start

    def _robust_centroid(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Geometric-median estimate via T Weiszfeld/IRLS steps (detached).

        Returns a [hidden] location m that is robust to feature-space outliers.
        Detached from the graph: it serves as a stable robust *reference point*;
        gradients flow through the final trimming weights and the pooled vector,
        not through the fixed-point recurrence (keeps optimisation well-behaved
        and the estimate a true location, like a58's IRLS reference).
        """
        with torch.no_grad():
            m = h.mean(dim=0)                                  # [hidden] init = mean
            for _ in range(self.irls_steps):
                d = torch.norm(h - m, dim=1)                   # [N] distance to m
                u = 1.0 / (1.0 + (d / r).pow(2))               # [N] Cauchy weight
                u = u / u.sum().clamp(min=1e-8)                # normalise (sum=1)
                m = (u.unsqueeze(1) * h).sum(dim=0)            # [hidden] reweighted
        return m

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract).
        h = self.bottleneck(features)                          # [N, hidden]
        n = h.size(0)

        if self.robust and n > 1:
            r = F.softplus(self.log_r) + 1e-3                  # >0 Weiszfeld scale
            s = F.softplus(self.log_s) + 1e-3                  # >0 trim temperature
            m = self._robust_centroid(h, r)                    # [hidden] robust centre
            d2 = ((h - m) ** 2).sum(dim=1)                     # [N] squared distance
            # Soft-trimmed weights: closer-to-centre patches weigh more.
            w = torch.softmax(-d2 / s, dim=0)                  # [N] sum=1
            z = (w.unsqueeze(1) * h).sum(dim=0)                # [hidden] robust pool
        else:
            # a88 ablation (robust=False) or degenerate N==1: PLAIN MEAN POOL.
            z = h.mean(dim=0)                                  # [hidden] diffuse mean

        y = self.head(z).view(-1)                              # [1] RAW grade logit
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, None, None
        return y, None, None


def _inv_softplus(x: float) -> float:
    """Inverse softplus so softplus(_inv_softplus(x)) == x for x > 0."""
    import math
    x = float(max(x, 1e-4))
    # log(exp(x) - 1), numerically stable for moderate x.
    return math.log(math.expm1(x)) if x < 20 else x


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    robust=True,           # a87 main = outlier-robust soft-trimmed diffuse pool
    irls_steps=3,
    scale_init=1.0,
    trim_init=4.0,
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    clamp_output=False,    # RAW logits out; trainer rounds+clips at eval
)
