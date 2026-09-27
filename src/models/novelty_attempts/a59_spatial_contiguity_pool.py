"""a59 — Consensus-consistency (spatial-contiguity) severity pooling.

Angle (spatial contiguity WITHOUT coordinates).
Reticulin fibrosis is a *connected meshwork*: the fibre signal is spatially
contiguous and diffuse across the marrow, not a property of a few isolated
patches. The bag exposes grid coords `rc=[N,2]` in the .pt, but the trainer
passes ONLY `features` to forward (no rc at runtime), so we cannot use the
coordinates directly. Instead we exploit contiguity with a *permutation-
invariant smoothness prior*: a patch that is CONSISTENT with the bag-wide
consensus is part of the coherent tissue signal and should count more; an
isolated, off-consensus patch is likely a noisy/edge artefact and is
down-weighted. The "consensus" is the bag's own encoded centroid, recomputed
per bag from the data — there is no fixed/learned direction.

Mechanism:
    h_i = Dropout(ReLU(Linear(1280->128) f_i))          # baseline encoder
    c   = mean_i h_i                                     # bag consensus (centroid)
    sim_i = cos(h_i, c)                                  # agreement w/ consensus in [-1,1]
    w_i = softmax_i( sim_i / temperature )               # consistency weight (sum=1)
    g_i = Linear(32->1)(ReLU(Linear(128->32) h_i))       # per-patch NONLINEAR severity
    y   = clamp( sum_i w_i * g_i , 0, 3)                 # consistency-weighted severity

Why this could beat the gated-attention baseline (0.7888):
  - The baseline's attention scores each patch against a single LEARNED query
    (gated V*U->W). It can lock onto a few high-attention patches and is blind
    to whether the signal is a coherent meshwork or scattered noise. Here the
    weight measures agreement with the *bag's own consensus*, so it rewards the
    diffuse, mutually-consistent fibre signal and suppresses lone outliers —
    matching how a connected reticulin meshwork actually looks. The reference
    is data-driven (the centroid), not a fixed axis, so it adapts per bag.
  - It keeps the distribution-aware per-patch NONLINEAR severity (a56's active
    ingredient) but replaces the parameter-free unweighted mean with a
    parameter-free consensus-consistency weight (no extra learned scorer
    params), staying low-capacity.

Why NOT a dead-end (explicit checks against the refuted list):
  - NOT coverage/extent/fraction (a52/a54, refuted 3x): no threshold tau, no
    sigmoid-fraction readout; g_i is an unbounded severity and the bag readout
    is a weighted mean of severities.
  - NOT a frozen/learned single-axis attention scorer (a25/a49/a50, "score-
    based softmax over a DIRECTION"): there is NO direction parameter. The
    softmax is over cosine-to-CONSENSUS, where consensus = mean_i h_i is
    recomputed per bag — a coherence prior, not a projection onto a fixed or
    learned vector. No attention_V/U/W query head at all.
  - NOT a sigmoid/softmax GATE between the bag rep and the scalar (DE11-13,
    HARD constraint): the softmax is a per-patch weighting computed BEFORE
    aggregation; the readout is the weighted mean of g_i then clamp — linear,
    unbounded, no gate multiplying the final scalar.
  - NOT top-k/argmax (a11/a12): every patch contributes a soft weight; nothing
    is selected or dropped.
  - NOT attention+mean concat (a13), not mean+std/quantile concat, not a
    prediction-level blend: single weighted-mean readout, one scalar.
  - NOT ||h||-norm prior (a45): the weight uses cosine similarity (direction in
    feature space), which is magnitude-invariant; ||h_i|| never enters.
  - DeepSet honesty: this is a low-capacity phi-(consensus-weighted-mean)-rho
    set function. phi is the tiny 128->32->1 severity head; the weighting adds
    ZERO learned params (only a temperature scalar). DE01 found DeepSet overfit
    at HIGH capacity on the legacy baseline; here total params stay < baseline.

Permutation-invariant: consensus = mean is permutation-invariant; softmax over
the per-patch sims and the weighted sum are permutation-equivariant->invariant.
Bag-size-invariant: duplicating the bag leaves c unchanged, leaves each sim_i
unchanged, and rescales every softmax logit's partition identically, so the
weighted-mean severity is unchanged (up to numerical noise).

Ablation companion (recommended): a60 = same model with the consistency weight
replaced by a uniform 1/N weight (i.e. == a56 unweighted mean). a59 vs a60
isolates the active ingredient = "does consensus-consistency weighting beat the
plain mean of per-patch severities?".

Kill criterion: abandon if a59 val_qwk < 0.78 at seed=2 AND a59 <= a60
(consensus weighting adds nothing over the plain severity mean). DoD = multi-
seed audit {0,1,2,3,42}, not seed=2 alone.

Param count (input_dim=1280, hidden=128, sev_hidden=32):
    bottleneck Linear(1280,128)+b = 163,968
    severity   Linear(128,32)+b   =   4,128
               Linear(32,1)+b     =      33
    log_temp scalar               =       1
    --------------------------------------------
    total                         = 168,130   (< 197,250 baseline)
"""
from __future__ import annotations

import math
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
        temperature: float = 0.1,        # softmax temp on cosine-to-consensus
        weight_mode: str = "consensus",  # "consensus" (a59) | "uniform" (a60 ablation)
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        assert weight_mode in ("consensus", "uniform"), weight_mode
        assert temperature > 0, "temperature must be positive."
        self.weight_mode = weight_mode
        self.clamp_output = clamp_output

        # Baseline-identical encoder (capacity lives here).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-patch NONLINEAR severity head (distribution-aware; a56's lever).
        self.severity = nn.Sequential(
            nn.Linear(hidden_dim, sev_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(sev_hidden, 1),
        )
        # Start mid-grade so the weighted severity sits in the interior of [0,3].
        with torch.no_grad():
            self.severity[-1].bias.fill_(1.5)

        # Learnable softmax temperature for the consistency weight (1 scalar,
        # parameterised in log-space to stay positive). NOT a per-patch scorer.
        self.log_temp = nn.Parameter(torch.tensor(math.log(float(temperature))))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag; batch_size=1 trainer contract)
        h = self.bottleneck(features)                 # [N, hidden]
        g = self.severity(h).squeeze(-1)              # [N] per-patch severity

        n = h.size(0)
        if self.weight_mode == "consensus" and n > 1:
            # Bag consensus = data-driven centroid (no learned direction).
            c = h.mean(dim=0, keepdim=True)           # [1, hidden]
            # Cosine agreement with consensus (magnitude-invariant: no ||h||).
            sim = F.cosine_similarity(h, c, dim=1)    # [N] in [-1, 1]
            temp = self.log_temp.exp().clamp(min=1e-3)
            w = F.softmax(sim / temp, dim=0)          # [N], sum=1
        else:
            # Uniform weight -> plain mean of per-patch severities (a60 ablation
            # / single-patch fallback). Keeps it permutation- & size-invariant.
            w = torch.full_like(g, 1.0 / max(n, 1))

        y = (w * g).sum().view(1)                     # consistency-weighted severity
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, w, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    sev_hidden=32,
    temperature=0.1,
    weight_mode="consensus",   # a59 main = consensus-consistency weighting
    clamp_output=True,
)
