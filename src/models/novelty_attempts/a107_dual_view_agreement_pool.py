"""a107 — DUAL-VIEW sign-AGREEMENT diffuse pool (one backbone, no fusion infra).

LENS = MULTI-VIEW. We read the SAME Virchow2 bag through TWO complementary,
fully DETERMINISTIC views of the per-patch fibrosis score and keep ONLY the
grade evidence that is corroborated by BOTH views. The two views are:

  (A) MEAN view  : m_raw = mean_i <h_i, w>          (absolute diffuse density;
                                                     sensitive to outlier patches)
  (B) MEDIAN view: m_rnk = rank-median_i <h_i, w>   (SAME score, SAME units/sign;
                                                     an order-statistic location
                                                     robust to outlier patches)

Both come from the SAME bottleneck embedding h and the SAME projection w — there
is exactly ONE backbone, ONE encoder, ONE direction. The two views are two
estimates of the SAME diffuse density scalar: the arithmetic MEAN and the
rank-MEDIAN of the per-patch score. They are deliberately complementary order
statistics — the mean is efficient but a few extreme artefact patches (lone bone
fragment, stain/edge artefact with a huge |score|) can drag it across zero; the
median is robust and stays put. Evidence that is high (or low) in BOTH views is
real diffuse fibrosis; evidence high in only the mean is an outlier-patch
artefact. We keep ONLY the corroborated (sign-agreeing) part.

WHY THIS IS NOT a79 (multi-mechanism ENSEMBLE) NOR an average -> tied mean
--------------------------------------------------------------------------
a79 builds three INDEPENDENT heads and does y = (y_att + y_mean + y_cov)/3. That
is an additive average: at convergence each head re-fits the SAME target and the
sum just rescales to the baseline mean -> tied paired delta (documented).

a107 NEVER adds the two views. It pools each view to a bag-level diffuse mean and
then combines them by a SIGNED-AGREEMENT (gated geometric-mean-like) operator:

    m_raw  = mean_i s_raw_i                  # diffuse density, raw view  (scalar)
    m_rnk  = mean_i s_rnk_i                  # diffuse density, rank view (scalar)

    agree  = relu(m_raw) * relu(m_rnk) - relu(-m_raw) * relu(-m_rnk)
           = sign-consistent product of the two views (an AND, not a mean)

    y = w_raw * m_raw  +  gamma * agree  +  b

`agree` is a PRODUCT of the two views. It is large-positive only when BOTH views
agree the bag is high-grade, large-negative only when BOTH agree low-grade, and
~0 whenever the two views DISAGREE (one positive, one negative, or either near
zero). A product cannot be reproduced by any fixed linear average of the two
means: d(agree)/d(m_raw) = sign-gated * m_rnk depends on the OTHER view, so the
combined readout is NOT in the span of {m_raw, m_rnk, const}. Concretely, on a
bag where m_raw and m_rnk have OPPOSITE signs (raw says high because of a global
gain shift, rank says the within-bag distribution is actually flat/low), `agree`
is driven to ~0 and CANCELS the spurious raw evidence — something no convex
average of the two can do. That is the "reinforced where robust, suppressed where
view-specific" behaviour the lens asks for, and it is why it does not degenerate
to a tied mean.

WHY THIS IS NOT a68 (backbone FUSION)
-------------------------------------
a68 concatenates TWO backbones (Virchow2 1280 + UNI2-h 1536) and needs the
2816-d fused cache + feature_dim_override. a107 uses ONE backbone, the standard
1280-d Virchow2 bag the trainer already passes; the "two views" are two
deterministic transforms of the SAME features. No fusion cache, no second
encoder, input_dim stays 1280.

WHY THIS IS NOT a relabel of a58/a87 (robust median/trimmed) or a43/a81
-----------------------------------------------------------------------
This is the most important distinctness check, stated honestly:
  - a58 takes a robust soft-MEDIAN of per-patch severities and a87 a robust
    feature centroid — each SUBSTITUTES one robust location estimator FOR the
    mean and pools that single estimator. Both tied, because a single location
    (mean OR median OR trimmed) is still ONE diffuse readout. a107 does NOT
    substitute: it KEEPS the mean view and ADDS the median view, then commits
    only to their SIGN-AGREEMENT PRODUCT. The load-bearing object is the
    interaction between two locations, not either location alone. Crucially the
    ablation a108 collapses to the MEAN baseline (gamma=0), NOT to the median —
    so a107 vs a108 isolates exactly "does cross-checking the mean against the
    median (and vetoing where they disagree) beat the mean alone?", a question
    no single-estimator attempt could pose.
  - a43/a81 each REPLACE the raw bag with one transformed view (norm-rank
    weighting / unit-direction) and pool that ONE view. a107 never re-weights a
    patch and never removes magnitude; both views read the SAME un-reweighted
    scores. Norm ||h|| is never used to weight anything (grading: no ||h||).

ESCAPING THE TWO DOCUMENTED TRAPS
---------------------------------
(1) UNDERFIT trap (1.3K-param / 6-param / 2-param readouts underfit, val 0-0.77):
    a107 keeps the FULL baseline-capacity encoder Linear(1280->128)+ReLU+Dropout
    (164K params) — the same encoder that puts mean-pool / gated-attn at the
    U-shape optimum (0.827/0.959). The diffuse mean of a 128-d learned embedding
    is NOT a low-DOF quantile/coverage readout; it has the same expressive power
    as the tied baseline. So it does not sit in the underfit valley.

(2) VAL-OVERFIT trap (richer structured mechanisms at baseline capacity lift
    seed=2 val but TANK test -> paired delta <= 0; a91/a93/a99):
    The ONLY learnable degrees of freedom a107 adds OVER the plain diffuse-pool +
    linear-readout baseline are TWO SCALARS: gamma (>=0, the agreement gain) and
    w_raw (the raw-view gain) plus bias — there is no new high-dimensional object
    (no rank-r subspace basis, no spectral filter, no extra 128-d direction set,
    no second attention head). a93/a99 added structured representations whose many
    free directions over-fit the hard fold; a107 adds a single nonlinearity
    (a product of two means) governed by ONE extra scalar. The rank view itself
    is PARAMETER-FREE and DATA-DRIVEN (within-bag ordering), so it cannot acquire
    fold-specific directions. The agreement operator is a VARIANCE-REDUCING
    INTERSECTION: it can only SHRINK evidence toward zero when the two views
    disagree, never invent new fold-specific signal. That is the mechanism by
    which it should move the PAIRED delta (val<->test gap) rather than just the
    seed=2 val: on a fold where the raw view is fooled by a global gain shift, the
    rank view vetoes it; on a fold where a few outlier patches inflate the rank
    view, the raw mean keeps it honest. The veto fires on BOTH val and test, so
    the gain is paired, not a seed=2 lottery.

The agreement nonlinearity is bounded and Lipschitz in the two means (a product
of two ReLU'd quantities, each a mean of bounded per-patch scores), so it cannot
explode at the real Virchow2 feature scale and adds no NaN risk.

WARM-START (train-only seed=2; init-only, no leakage; note a103 caveat)
-----------------------------------------------------------------------
The projection w is initialised from the bottleneck's response to the train-only
fibrosis axis (data/prototypes_virchow2_reti_train_seed2.pt['axis']), exactly the
a83/a105 idiom — a pure INITIALISATION, then fully learned. w_raw starts at 1,
gamma starts SMALL (so a107 at init == the warm diffuse-mean baseline and the
agreement term is earned, not assumed), bias starts at mid-grade 1.5. Per the
a103 finding, warm-start mainly helps the seed=2 fold and washes out cross-fold;
a107 does NOT depend on it for its paired claim — the agreement veto is the
load-bearing ingredient and is independent of the warm direction (set warm_start
=False to confirm the veto alone carries the delta).

PERMUTATION- & BAG-SIZE-INVARIANCE / DETERMINISM
------------------------------------------------
- s_raw is per-patch; mean over patches is permutation-invariant and N-normalised.
- The rank-z view: rank within bag is permutation-equivariant, and its mean is a
  fixed deterministic function of N (the rank ramp is symmetric, mean exactly 0),
  so m_rnk is permutation-invariant and bag-size-invariant (the ramp is rescaled
  by N). Duplicating the bag leaves both means unchanged up to ties.
- No sampling, no dropout-at-inference effect (eval mode); fully deterministic.

ABLATION (a108): set gamma_fixed=0.0 and freeze gamma (no agreement term) ->
    y = w_raw * m_raw + b  == the plain warm diffuse-mean readout (the TIED
    baseline). a107 vs a108 removes EXACTLY the active ingredient — the dual-view
    sign-agreement product — with the bottleneck, w, w_raw and bias byte-identical.

PARAM COUNT (input_dim=1280, hidden=128):
    bottleneck Linear(1280,128)+b = 163,968
    w          [128]              =     128
    w_raw, gamma_logit, bias      =       3
    -----------------------------------------
    total                         = 164,099   (< 197,250 baseline; matches a83)
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
    """Unit train-only fibrosis axis (1280-d, seed=2). None if cache absent."""
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
        gamma_init: float = 0.1,        # initial agreement gain (small; earned)
        gamma_fixed: Optional[float] = None,  # a108 ablation: pin gamma (e.g. 0.0)
        warm_start: bool = True,        # init w along train fibrosis axis (init-only)
        prototype_path: Optional[str] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        self.eps = float(eps)
        self.gamma_fixed = gamma_fixed

        # Baseline-identical encoder (this is where all the capacity lives).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Single shared per-patch fibrosis direction (BOTH views read it).
        self.w = nn.Parameter(torch.randn(hidden_dim) / (hidden_dim ** 0.5))

        # Raw-view gain and bias (the baseline diffuse-mean readout).
        self.w_raw = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(1.5))  # mid-grade start

        # Agreement gain gamma >= 0 via softplus(gamma_logit). When pinned
        # (ablation), gamma is a fixed buffer and contributes no gradient.
        if gamma_fixed is None:
            g = max(float(gamma_init), 1e-4)
            # softplus^{-1}(g): logit s.t. softplus(logit) == g
            inv = torch.log(torch.expm1(torch.tensor(g)).clamp(min=1e-8))
            self.gamma_logit = nn.Parameter(inv)
        else:
            self.register_buffer("gamma_const", torch.tensor(float(gamma_fixed)))

        # Warm-start the projection toward the bottleneck response to the
        # train-only fibrosis axis (init-only; then fully learned).
        if warm_start:
            path = Path(prototype_path) if prototype_path else _DEFAULT_PROTOTYPE_PATH
            axis = _load_fibrosis_axis(path, input_dim)
            if axis is not None:
                with torch.no_grad():
                    resp = self.bottleneck[0].weight.detach() @ axis  # [hidden]
                    resp = resp / resp.norm().clamp(min=self.eps)
                    self.w.copy_(resp)

    def _gamma(self) -> torch.Tensor:
        if self.gamma_fixed is None:
            return F.softplus(self.gamma_logit)
        return self.gamma_const

    @staticmethod
    def _rank_location(s: torch.Tensor) -> torch.Tensor:
        """Within-bag RANK-MEDIAN robust location of the per-patch score s [N].

        This is the SECOND deterministic, parameter-free VIEW of the SAME diffuse
        density m_raw = mean(s): a rank-based (order-statistic) location estimate
        of the same scalar, on the SAME scale and SIGN, but robust to a handful
        of extreme outlier patches. We use the median (true rank-50 order
        statistic, interpolated for even N) rather than the mean. It is:
          - SIGNED and scale-comparable to m_raw (same units), so the two views
            can genuinely DISAGREE IN SIGN -> the agreement veto has both branches
            live (unlike a non-negative spread/scatter statistic).
          - permutation-INVARIANT (order statistic) and bag-size-INVARIANT
            (a quantile of the within-bag distribution, not a sum).
          - the OPPOSITE failure mode to the mean: a few artefact patches with
            huge |score| swing the MEAN across zero but leave the MEDIAN put. So
            m_raw (mean) and m_rnk (median) disagree in sign exactly when outlier
            patches are corrupting the absolute-density view -> the case the veto
            must suppress.
        Deterministic; no learnable parameter; no patch is re-weighted.
        """
        n = s.numel()
        if n <= 1:
            return s.reshape(-1)[0] if n == 1 else s.new_zeros(())
        s_sorted = torch.sort(s).values
        mid = (n - 1) / 2.0
        lo = int(mid)
        hi = min(lo + 1, n - 1)
        frac = mid - lo
        return s_sorted[lo] * (1.0 - frac) + s_sorted[hi] * frac  # interp median

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: ONE bag [N, input_dim] (single-bag, batch_size=1 contract).
        h = self.bottleneck(features)                       # [N, hidden]
        s_raw = h @ self.w                                  # [N] per-patch raw score

        # VIEW A: diffuse density on the absolute scale.
        m_raw = s_raw.mean()                                # scalar

        # VIEW B: the within-bag RANK-MEDIAN robust location of the SAME score.
        # Same units / same sign as m_raw, but an order-statistic location that a
        # handful of extreme artefact patches cannot drag across zero. The two
        # views are two estimates of the SAME diffuse density that DISAGREE IN
        # SIGN exactly when outlier patches corrupt the absolute-mean view.
        m_rnk = self._rank_location(s_raw)                  # scalar (param-free)

        # SIGN-AGREEMENT product (an AND of the two views, NOT an average).
        # +m_raw*m_rnk when both say high, +(-m_raw)*(-m_rnk) when both say low,
        # ~0 whenever the mean view and the median view disagree in sign. A
        # product, so it is NOT in the linear span of {m_raw, m_rnk, const}.
        agree = (
            F.relu(m_raw) * F.relu(m_rnk)
            - F.relu(-m_raw) * F.relu(-m_rnk)
        )

        gamma = self._gamma()
        y = (self.w_raw * m_raw + gamma * agree + self.bias).view(1, 1)

        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    gamma_init=0.1,
    gamma_fixed=None,      # a108 ablation: set to 0.0 to remove the agreement term
    warm_start=True,
    prototype_path=None,   # None -> data/prototypes_virchow2_reti_train_seed2.pt
    eps=1e-6,
)
