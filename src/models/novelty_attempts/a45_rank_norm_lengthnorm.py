"""a45 — rank-norm soft-mean + length-normalised temperature (H22 main, batch 22).

Philosophy bucket: norm_based_salience

Hint targeted (NOVELTY_SEARCH_PLAYBOOK.md §12.4, post-mortem of a01–a44):
    norm_based_salience family — extended a44 with bag-size-adaptive
    rank-weighting temperature, motivated by the empirical observation
    that bag sizes in this dataset span 13–112 patches (n=1330,
    median 40, mean 44, p10=31, p90=48; measured on
    `data/features_virchow2_reti/`). The spread is ~9× between the
    smallest and largest bag — enough that a fixed τ allocates very
    different effective bandwidth across them.

Background
----------
a44_rank_norm_softmean_bottleneck reached val_qwk 0.798 / test_qwk 0.958
(test gate passes, val gate misses by 0.020) using a FIXED temperature
τ = 8.0 inside `softmax(-rank / τ)`. With FIXED τ, the effective number
of "high-weight" patches is the same regardless of bag size N. In small
bags this concentrates weight on just the few highest-norm patches; in
large bags the same few patches are diluted by hundreds of mid-norm
neighbours that all receive non-negligible weight.

Hypothesis (H22_rank_norm_lengthnorm_tau, refined from H21)
-----------------------------------------------------------
Replace τ = constant with **τ = c₀ · √N** so the softmax effective
width scales with bag size. Mechanism:
    w_i = softmax(-rank_i / (c₀ · √N))_i
With this τ, the *fraction* of the bag that receives meaningful weight
is held roughly constant across bag sizes — large bags get more patches
contributing, small bags stay sharp. c₀ is a learnable softplus
parameter so the model can re-discover the constant-τ behaviour if
length-norm hurts.

Rationale tying to a17/H10
--------------------------
The legacy H10 stack (a17_coverage_length_norm) showed that
length-normalised softmax temperature is one of the few things that
moved val_qwk in the attention_augment bucket (a17 val 0.7970 was the
ceiling of that bucket). a17 applied length-norm to LEARNED attention
scores; here we apply the same principle to the parameter-free
RANK-BASED scores from a44. This is a true compound across two
philosophy buckets (attention_augment → norm_based_salience) and is
therefore non-redundant with either parent.

Mechanism (exact)
-----------------
1. Input features [N, D=1280] (Virchow2).
2. Bottleneck   h_i = Dropout(ReLU(W_b h_i^raw))   ∈ R^128         (a44 design)
3. Rank by ||h||₂ descending   →   rank_i ∈ {0, …, N-1}
4. Temperature τ = softplus(c_raw) · √N            (length-normalised)
   - c_raw initialised so τ ≈ 8.0 when N = 40 (the median bag size
     under the locked Virchow2 reti split).
   - i.e. softplus(c_raw)|_init = 8.0 / √40 ≈ 1.2649
     →   c_raw_init = log(e^1.2649 − 1) ≈ 0.9335
5. w_i = softmax(-rank_i / τ)_i
6. bag = Σ_i w_i · h_i                              ∈ R^128
7. ŷ = Linear(128, num_classes)(bag).
   - num_classes=1 (scalar regression): clamp to [0, 3].
   - num_classes=4 (multi-class CE):    raw logits, no clamp.

Ablation companion
------------------
a46_rank_norm_constant_tau (Philosophy bucket: norm_based_salience).
Identical wiring with τ = softplus(c_raw) (NO √N factor). This is the
a44-equivalent inside the new compound family, retagged with the new
philosophy bucket. The a45 ↔ a46 contrast isolates the **length-norm
temperature** as the active ingredient on top of a44.

Kill criterion
--------------
Family-level: if BOTH a45 and a46 have val_qwk < 0.80 at seed=2,
declare H22 dead and the `norm_based_salience` compound bucket
exhausted; recommendation: move to `ordinal_head` per §12.4.
Main-only: if a45 ≤ a46 by ≥ 0.005 val_qwk, the length-norm temperature
is not the active ingredient — discard direction even if numbers are OK.

Param count (input_dim=1280, hidden_dim=128)
--------------------------------------------
    bottleneck Linear(1280, 128) + bias    = 163,968
    classifier Linear(128, num_classes) +b = 129 (scalar) / 516 (4-class)
    c_raw      learnable scalar            =       1
    -------------------------------------------------
    total trainable (scalar)               = 164,098
    total trainable (4-class)              = 164,485
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
        # τ ≈ 8.0 at N = 40 (median bag size in features_virchow2_reti),
        # matching a44's fixed τ at the median.
        c_raw_init: float = 0.9335,  # softplus(0.9335) ≈ 1.2649 ≈ 8/√40
        length_norm: bool = True,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes in (1, 4), (
            "a45 supports num_classes=1 (scalar regression) "
            "or num_classes=4 (multi-class classification)."
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.c_raw = nn.Parameter(torch.tensor(float(c_raw_init)))
        self.length_norm = length_norm
        self.clamp_output = clamp_output
        self.is_scalar = (num_classes == 1)

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

        # Rank by bottlenecked-feature L2 norm
        norms = h.norm(dim=-1)  # [B, N]
        order = norms.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1, order, torch.arange(N, device=features.device).expand(B, N)
        )

        # Length-normalised temperature: τ = softplus(c_raw) · √N
        base = F.softplus(self.c_raw).clamp(min=1e-3)
        tau = base * math.sqrt(N) if self.length_norm else base

        w = torch.softmax(-ranks.float() / tau, dim=1)  # [B, N]

        bag = (w.unsqueeze(-1) * h).sum(dim=1)  # [B, H]
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
    c_raw_init=0.9335,
    length_norm=True,
    clamp_output=True,
)

