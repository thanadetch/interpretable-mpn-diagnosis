"""a51 — RNL-MIL + middle-mass complement (H23 main, batch 25, G1-fix attempt).

Philosophy bucket: norm_based_salience

Background (post-mortem of a45 N=10 paired audit, 2026-05-27)
-------------------------------------------------------------
a45_rank_norm_lengthnorm produced a defensible per-seed win:
    QWK : 5/10 joint wins, Δval median +0.005, Δtest median +0.015
    Acc : Δ mean +4.30, Δ median +4.20
    mR  : Δ mean +2.34, Δ median +3.34 (a45 wins 7/10 seeds)
    G0  : Δ mean +10.43, wins 6/10
    G3  : Δ mean +2.76,  wins 5/10
    G2  : Δ mean +0.89,  wins 5/10
    G1  : Δ mean -4.70,  loses 7/10  ← only persistent regression

The aggregator's rank-by-norm softmax (τ = c·√N) concentrates weight on
the few patches with the highest bottlenecked-feature L2 norm — pathologically
corresponding to the most fibre-positive patches. This is desirable for
the extreme grades (G0 = "no fibre anywhere" / G3 = "fibre dense everywhere"),
but G1/G2 require integrating BOTH fibre-positive AND fibre-negative
evidence in calibrated proportions, which the rank-top focus suppresses.

Hypothesis (H23_rank_norm_mid_mass)
-----------------------------------
Add a **complementary middle-mass pool**: the mean of patches whose rank
falls in the central [N/4, 3N/4] band (i.e. dropping the top-25% and
bottom-25%). Mid-rank patches carry the "some-fibre-but-not-dominant"
evidence that intermediate grades need. The bag representation becomes:

    bag = α · rank_pool + (1 - α) · mid_pool

with α = sigmoid(α_raw), α_raw initialised at logit(0.85) ≈ 1.7346 so the
model starts strongly rank-dominant (close to a45). If mid-mass evidence
turns out useless, α drifts back to 1.0 and a51 reduces to a45.

Mechanism (exact)
-----------------
1. Bottleneck h_i = Dropout(ReLU(W_b · h_i^raw))   ∈ R^128   (a45-identical)
2. Rank by ||h_i||_2 descending; let r_i ∈ {0..N-1}.
3. τ = softplus(c_raw) · √N                          (a45-identical)
4. **Path A** (rank pool, a45-identical):
       w_i = softmax(-r_i / τ)
       rank_pool = Σ_i w_i · h_i                    ∈ R^128
5. **Path B** (mid-mass, NEW):
       m_i = 1 if N/4 ≤ r_i < 3N/4 else 0
       mid_pool = (Σ_i m_i · h_i) / max(Σ m_i, 1)   ∈ R^128
       (For N < 4 the mask collapses; we fall back to mean of all patches.)
6. α = sigmoid(α_raw)                                 ∈ (0, 1)
7. bag = α · rank_pool + (1 - α) · mid_pool          ∈ R^128
8. ŷ = Linear(128, num_classes)(bag).
   - num_classes=1 (scalar regression): clamp to [0, 3].
   - num_classes=4 (multi-class CE):    raw logits, no clamp.

Ablation companion (not in this file)
-------------------------------------
a52_mid_mass_only (TODO if needed): α forced to 0 — pure mid-mass mean.
Tests whether the COMBINATION (a51) or mid-mass alone is the active
ingredient. Skip unless a51 itself shows promise.

Kill criterion (pre-registered)
-------------------------------
Goal: a51 must beat a45 on **macro recall AND G1 recall** at the same
time, without destroying G3 recall. Acceptable outcomes:
  WIN:  median Δ(a51 - a45) macro_recall > +1.0  AND  median Δ G1 > +2.0
        AND   median Δ G3 ≥ -2.0   (over 5 paired seeds)
  TIE:  any of the above misses by < 1.0
  LOSE: median Δ macro_recall ≤ 0  OR  destroys G3 by ≥ 2.0
        → discard, lock a45 as thesis aggregator.

Hard stop: ≤ 1/5 seeds where a51 ≥ a45 on macro recall → close H23.

Param count (input_dim=1280, hidden_dim=128)
--------------------------------------------
    bottleneck Linear(1280, 128) + bias    = 163,968
    classifier Linear(128, num_classes) +b = 129 (scalar) / 516 (4-class)
    c_raw      learnable scalar            =       1
    α_raw      learnable scalar            =       1  (NEW vs a45)
    -------------------------------------------------
    total trainable (scalar)               = 164,099   (+1 vs a45)
    total trainable (4-class)              = 164,486   (+1 vs a45)
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
        c_raw_init: float = 0.5413248,        # a45-identical → τ ≈ 8.0 at N=64
        alpha_raw_init: float = 1.7346,       # sigmoid(1.7346) ≈ 0.85 (rank-dominant)
        length_norm: bool = True,
        clamp_output: bool = True,
        mid_low: float = 0.25,                # mid-mass band lower fraction
        mid_high: float = 0.75,               # mid-mass band upper fraction
    ) -> None:
        super().__init__()
        assert num_classes in (1, 4), (
            "a51 supports num_classes=1 (scalar regression) "
            "or num_classes=4 (multi-class classification)."
        )
        assert 0.0 <= mid_low < mid_high <= 1.0, "mid-band must satisfy 0 ≤ low < high ≤ 1."
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.c_raw = nn.Parameter(torch.tensor(float(c_raw_init)))
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_raw_init)))
        self.length_norm = length_norm
        self.clamp_output = clamp_output
        self.mid_low = mid_low
        self.mid_high = mid_high
        self.is_scalar = (num_classes == 1)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        B, N, _ = features.shape

        # Bottleneck per-patch
        h = self.bottleneck(features.reshape(B * N, -1)).reshape(B, N, -1)  # [B, N, H]

        # Rank by bottlenecked-feature L2 norm (descending → high-norm = rank 0)
        norms = h.norm(dim=-1)  # [B, N]
        order = norms.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1, order, torch.arange(N, device=features.device).expand(B, N)
        )

        # Path A: rank-softmax pool (a45-identical)
        base = F.softplus(self.c_raw).clamp(min=1e-3)
        tau = base * math.sqrt(N) if self.length_norm else base
        w = torch.softmax(-ranks.float() / tau, dim=1)  # [B, N]
        rank_pool = (w.unsqueeze(-1) * h).sum(dim=1)    # [B, H]

        # Path B: mid-mass pool (mean of ranks in [N·mid_low, N·mid_high))
        low_k = int(math.floor(self.mid_low * N))
        high_k = int(math.floor(self.mid_high * N))
        if high_k <= low_k:  # degenerate (very small bag): fall back to global mean
            mid_pool = h.mean(dim=1)
        else:
            mid_mask = (ranks >= low_k) & (ranks < high_k)  # [B, N]
            mid_mask_f = mid_mask.float()
            denom = mid_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            mid_pool = (mid_mask_f.unsqueeze(-1) * h).sum(dim=1) / denom  # [B, H]

        # Convex combine
        alpha = torch.sigmoid(self.alpha_raw)  # scalar in (0, 1)
        bag = alpha * rank_pool + (1.0 - alpha) * mid_pool  # [B, H]

        y = self.classifier(bag)  # [B, num_classes]
        if self.is_scalar and self.clamp_output:
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
    alpha_raw_init=1.7346,
    length_norm=True,
    clamp_output=True,
    mid_low=0.25,
    mid_high=0.75,
)

