"""a58 — Robust (soft-median / IRLS-Huber) pooling of per-patch severities.

Hypothesis (Hn_robust_severity). A reticulin ROI is graded by the DIFFUSE
fibre density of the *marrow*, but every ROI also contains a few non-marrow
patches — bone trabeculae, fat, edge/section artefacts — whose learned
fibrosis-severity g_i is an OUTLIER (often spuriously high or low). The plain
mean of {g_i} (a56) lets those few outliers drag the bag estimate. The
clinically-faithful summary is the *typical* (central) severity of the marrow,
i.e. a ROBUST LOCATION of the severity distribution, not its arithmetic mean.

    h_i = Dropout(ReLU(Linear(1280->128) f_i))      # baseline encoder
    g_i = Linear(32->1)(ReLU(Linear(128->32) h_i))  # per-patch severity (NONLINEAR)
    y   = clamp( soft_median({g_i}), 0, 3 )          # ROBUST aggregate (not the mean)

The robust aggregate is a differentiable M-estimator of location computed by a
FIXED number of IRLS (iteratively-reweighted least-squares) steps with a
Huber/Welsch influence function:

    m_0 = mean_i g_i                                  # init at the mean
    repeat T times:
        r_i  = g_i - m_t
        w_i  = 1 / (1 + (r_i / c)^2)                  # Welsch-style robust weight
        m_{t+1} = sum_i w_i g_i / sum_i w_i           # reweighted location
    y = clamp(m_T, 0, 3)

`c` (a learnable, softplus-positive scale) sets how far a patch may deviate
before it is down-weighted as an outlier. As c -> inf this collapses to the
plain mean (so a56 is strictly nested inside a58 as the no-robustness limit);
for finite c it converges toward the soft-MEDIAN of the severity distribution.
The weights are RESIDUAL-based (how far g_i is from the current central
estimate) — there is NO direction, NO threshold, NO fraction, NO norm, NO
attention over features.

Why this is grading-aligned & genuinely different:
  - It reads the CENTRAL severity of the marrow, robust to a minority of
    bone/fat/artefact patches whose g_i is extreme. Mean-pool (a53) and
    mean-of-severities (a56) cannot do this — one very-high or very-low patch
    shifts their answer linearly. The robust location is, by construction,
    insensitive to a few outliers.
  - The map g -> soft_median(g) is permutation-invariant (only sums over i)
    and bag-size-invariant (every m_{t+1} is a normalized weighted average, so
    duplicating the bag leaves all w_i and m_t unchanged).

Why this is NOT a refuted dead-end:
  - NOT coverage/extent/fraction (a01/a02/a09/a52/a54, refuted 3x): the
    aggregate is a robust LOCATION of unbounded severities, never a count of
    "how many patches exceed a threshold". No sigmoid-of-score, no fraction.
  - NOT the plain mean of severities (a56) NOR mean-pool (a53): the IRLS
    weights make this a NONLINEAR-in-the-bag M-estimator; a56 is the c->inf
    special case (the ablation companion below makes that explicit).
  - NOT top-k / argmax / trimmed / median-of-raw (a11/a12, DE06): DE06 trimmed
    RAW features with a HARD, hand-set fraction; here we softly down-weight by
    a SMOOTH residual influence on LEARNED severities, fully differentiable and
    with a single learnable scale c (no hard cutoff, no fixed trim fraction,
    no hard rank selection — every patch keeps a continuous weight in (0,1]).
  - NOT a sigmoid/softmax GATE between the bag rep and the scalar (DE11-15):
    the only nonlinearity feeding the scalar is the per-patch ReLU MLP
    (PRE-aggregation) and the residual reweighting (which is location-only,
    not a multiplicative gate on a bag representation). The readout is the
    robust location itself, linear/unbounded, clamped only at the very end.
  - NOT norm-based (a45): ||h|| is never used; weights depend only on the
    severity residual r_i = g_i - m_t.
  - NOT DeepSet/Set-Transformer at high capacity (DE01): phi is the tiny
    128->32->1 head, rho is the parameter-free IRLS soft-median; total extra
    params over the encoder are ~4.1K plus ONE scalar c.

Ablation companion: set robust=False (-> y = mean_i g_i, exactly a56) or send
c -> +inf. a58 vs a56 isolates the active ingredient: "does a ROBUST location
of per-patch severities beat their plain MEAN (i.e. do outlier bone/artefact
patches actually hurt the mean)?".

Kill criterion: abandon if a58 val_qwk < 0.78 at seed=2 AND a58 <= a56
(robustness adds nothing over the plain mean). DoD = multi-seed audit.

Closest dead-end (honest): DE06 trimmed/median pool. a58 is differentiable,
operates on LEARNED severities (not raw features), and learns its own scale c
rather than a hard trim fraction — but it lives in the same "ignore extreme
patches" family, so if DE06's refutation was really about "robust location
helps nothing here", a58 inherits that risk.

Param count (input_dim=1280, hidden=128, sev_hidden=32):
    bottleneck Linear(1280,128)+b = 163,968
    severity   Linear(128,32)+b    =   4,128
               Linear(32,1)+b      =      33
    robust scale (raw_c)           =       1
    -------------------------------------------
    total                          = 168,130   (< ~197,250 baseline; < 200K)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        sev_hidden: int = 32,
        robust: bool = True,        # True = a58 soft-median; False = a56 plain mean
        n_iter: int = 3,            # IRLS reweighting steps
        c_init: float = 0.5,        # initial robust scale (severity units)
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert n_iter >= 1
        self.robust = robust
        self.n_iter = int(n_iter)
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-patch NONLINEAR severity head (g_i = MLP(h_i)).
        self.severity = nn.Sequential(
            nn.Linear(hidden_dim, sev_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(sev_hidden, 1),
        )
        # Start mid-grade so the central severity sits in the interior of [0,3].
        with torch.no_grad():
            self.severity[-1].bias.fill_(1.5)

        # Learnable robust scale c > 0 (parameterised via softplus(raw_c)).
        # c_init -> raw_c so that softplus(raw_c) == c_init at start.
        c0 = float(c_init)
        raw0 = torch.log(torch.expm1(torch.tensor(c0)))  # inverse-softplus
        self.raw_c = nn.Parameter(raw0)

    def _soft_median(self, g: torch.Tensor) -> torch.Tensor:
        """Differentiable Welsch-IRLS robust location of severities g [N] -> [1].

        m_0 = mean(g); repeat: w = 1/(1+(r/c)^2); m = sum(w g)/sum(w).
        Permutation-invariant (sums over i) and bag-size-invariant (normalized
        weighted average). c -> inf reproduces the plain mean.
        """
        c = F.softplus(self.raw_c).clamp(min=1e-3)
        m = g.mean()
        for _ in range(self.n_iter):
            r = g - m
            w = 1.0 / (1.0 + (r / c) ** 2)      # Welsch-style influence in (0,1]
            m = (w * g).sum() / w.sum().clamp(min=1e-8)
        return m.view(1)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        h = self.bottleneck(features)             # [N, hidden]
        g = self.severity(h).squeeze(-1)          # [N] per-patch severity

        if self.robust:
            y = self._soft_median(g)              # robust central severity
        else:
            y = g.mean().view(1)                  # a56 ablation = plain mean

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Report the final IRLS weights as a per-patch "trust" map.
            if self.robust:
                c = F.softplus(self.raw_c).clamp(min=1e-3)
                with torch.no_grad():
                    m = g.mean()
                    for _ in range(self.n_iter):
                        r = g - m
                        w = 1.0 / (1.0 + (r / c) ** 2)
                        m = (w * g).sum() / w.sum().clamp(min=1e-8)
                attn = w / w.sum().clamp(min=1e-8)
            else:
                attn = g
            return y, attn, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    sev_hidden=32,
    robust=True,        # a58 main = robust soft-median of per-patch severities
    n_iter=3,
    c_init=0.5,
    clamp_output=True,
)
