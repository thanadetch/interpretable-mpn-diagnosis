"""a49 — learned-direction score soft-mean + length-norm temperature (H25 main, batch 24).

Philosophy bucket: norm_based_salience

Hint targeted (NOVELTY_NOTES.md frontmatter, post-mortem of a45/a46)
--------------------------------------------------------------------
H25_learned_direction_vs_norm_salience — a45/a46 confirmed that a
length-normalised softmax over a **scalar salience score** beats every
augment-style mechanism we have tried, but never isolated whether
||h|| is the *right* salience signal or merely a passable parameter-free
baseline. a49 / a50 close that gap with a learnable-direction scorer.

Design choice: score-based softmax (not rank-based)
---------------------------------------------------
a45 uses softmax over **ranks** of ||h||. Ranks are computed by
`argsort`, which is non-differentiable — so if a learnable direction
`v` were placed inside an "argsort and softmax-by-rank" pipeline, no
gradient would flow back to `v`. To let v actually learn, a49 replaces
the rank-of-salience with the salience score itself inside the softmax:

    s_i = <h_i, v>              (signed scalar projection)
    w_i = softmax(s_i / τ)      with τ = softplus(c_raw) · √N

This is the smooth, differentiable analogue of "rank by s_i then
softmax over rank". The √N temperature factor is preserved verbatim
from a45 (the active ingredient that survived the a46 ablation), so the
bag-size-adaptive sharpness is identical. The only structural difference
from a45 is that the per-patch salience driving the softmax is a learned
projection rather than the fixed L2 norm.

Mechanism (exact)
-----------------
1. Input features [N, D=1280] (Virchow2).
2. Bottleneck   h_i = Dropout(ReLU(W_b · h_i^raw))     ∈ R^128  (a45 design)
3. Direction v = v_raw / ||v_raw||_2                    ∈ R^128
   - v_raw ~ N(0, 1/√H) at init → init scale matches a Linear column.
   - Learnable in a49 (Parameter); frozen in a50 (Buffer, see a50.py).
4. Per-patch salience s_i = <h_i, v>                     ∈ R
5. Length-normalised temperature τ = softplus(c_raw) · √N
   - c_raw_init = log(e − 1) ≈ 0.5413 → softplus = 1.0 → τ ≈ 8.0 at N=64
     (same reference operating point as a44/a45).
6. Weights w_i = softmax(s_i / τ)_i                       ∈ Δ^{N-1}
7. Bag rep b = Σ_i w_i · h_i                              ∈ R^128
8. ŷ = clamp(Linear(128, 1)(b), 0, 3)

Sign convention: positive s_i → large weight (a45 is "low rank → large
weight"; the two are equivalent under monotone transformations).

Ablation companion
------------------
a50_random_direction_score. Identical wiring with v *frozen* at a random
unit vector drawn from N(0, I/H) with `random_direction_seed=2`. The
a49 ↔ a50 contrast isolates **"learned salience direction" vs
"random-direction projection"** as the active ingredient on top of
a45's parameter-free norm scorer.

Cross-batch comparison (the three-way table batch 24 resolves)
--------------------------------------------------------------
    a45 (rank by ||h||)        — frozen norm scorer (rank-softmax),    164,098 params
    a50 (score by random v)    — frozen random scorer (score-softmax), 164,098 params
    a49 (score by learned v)   — learned direction (score-softmax),    164,226 params

| outcome                              | interpretation                                                       |
|--------------------------------------|----------------------------------------------------------------------|
| a49 > a45 > a50                      | Direction matters AND learned > norm — new winner direction.          |
| a49 ≈ a50 > a45                      | Score-softmax > rank-softmax; direction itself is irrelevant.         |
| a45 > a49 ≈ a50                      | Rank structure carries a prior the score-softmax cannot match.        |
| a45 ≈ a49 ≈ a50                      | "Some scalar salience + √N softmax" is the active ingredient.         |

All four outcomes yield a publishable methodological finding about
parameter-free MIL salience.

Kill criterion
--------------
Family-level: declare H25 dead if BOTH a49 and a50 have val_qwk < 0.79
at seed=2.
Main-only: if a49 ≤ a50 by ≥ 0.005 val_qwk, the *learned* direction is
not the active ingredient; do not retry learned-direction salience
without a new mechanism.

Pathology rationale
-------------------
||h|| in bottleneck space conflates "amount of stain signal" with
"distance from origin"; a signed projection onto a learned v is a
**directional** salience that can encode "more like high-grade vs more
like low-grade" in a single scalar. This is the parameter-light analogue
of the H3 prototype-projection idea (DE22, killed at frozen prototypes
in raw 1280-d Virchow2 space) — but the learned direction in bottleneck
space sidesteps the DE22 failure mode (population mean underweights
informative orthogonal variation).

Param count (input_dim=1280, hidden_dim=128)
--------------------------------------------
    bottleneck Linear(1280, 128) + bias  = 163,968
    classifier Linear(128, 1) + bias     =     129
    v_raw      learnable R^128           =     128
    c_raw      learnable scalar          =       1
    -------------------------------------------------
    total trainable                       = 164,226
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
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        # τ = softplus(c_raw) · (√N if length_norm else 1). Init chosen so
        # τ ≈ 8.0 at N = 64 (median bag size), matching a45.
        c_raw_init: float = 0.5413248,  # softplus(0.5413) ≈ 1.0
        length_norm: bool = True,
        # Direction-scorer knobs. `learn_direction=True` (a49 main) makes v
        # a Parameter; False (a50 ablation) makes v a frozen Buffer drawn
        # from N(0, I/H) at construction time using `random_direction_seed`.
        learn_direction: bool = True,
        random_direction_seed: int = 2,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "a49 is scalar-regression only (num_classes=1)."
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.c_raw = nn.Parameter(torch.tensor(float(c_raw_init)))

        # Direction vector v ∈ R^H. Init scale matches a Linear weight
        # column (std = 1/√H) so gradients land in the same regime as the
        # bottleneck/classifier weights.
        init_std = 1.0 / math.sqrt(hidden_dim)
        gen = torch.Generator().manual_seed(int(random_direction_seed))
        v_init = torch.randn(hidden_dim, generator=gen) * init_std
        if learn_direction:
            self.v_raw = nn.Parameter(v_init)
        else:
            self.register_buffer("v_raw", v_init)

        self.length_norm = length_norm
        self.learn_direction = learn_direction
        self.clamp_output = clamp_output

    def _direction(self) -> torch.Tensor:
        return F.normalize(self.v_raw, dim=0, eps=1e-8)  # [H]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,  # signature parity with the trainer
        metrics: Optional[dict] = None,  # ignored
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)  # [1, N, D]
        B, N, _ = features.shape

        # Bottleneck per-patch
        h = self.bottleneck(features.reshape(B * N, -1)).reshape(B, N, -1)  # [B, N, H]

        # Per-patch signed salience: projection onto unit-norm v.
        v = self._direction()  # [H]
        s = h @ v  # [B, N]

        # Length-normalised temperature: τ = softplus(c_raw) · √N
        base = F.softplus(self.c_raw).clamp(min=1e-3)
        tau = base * math.sqrt(N) if self.length_norm else base

        # Score-based softmax (differentiable; routes gradient back to v).
        w = torch.softmax(s / tau, dim=1)  # [B, N]

        bag = (w.unsqueeze(-1) * h).sum(dim=1)  # [B, H]
        y = self.classifier(bag)  # [B, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if squeeze:
            y = y.squeeze(0)
            w = w.squeeze(0)
        return y, w, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    c_raw_init=0.5413248,
    length_norm=True,
    learn_direction=True,
    random_direction_seed=2,
    clamp_output=True,
)

