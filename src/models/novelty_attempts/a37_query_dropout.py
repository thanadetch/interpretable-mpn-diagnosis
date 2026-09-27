"""a37 - K=4 multi-query cross-attn + train-time per-query dropout (H19, batch 19 main).

Softer analogue of H18 (a35 orthogonality penalty, DE26): instead of
*forcing* the K=4 queries apart, randomly zero out individual query
bag-reps during training so the flatten+Linear head must be predictive
even when any single query is missing. At eval, all K queries are active
(no mask, no scaling) -> inference-identical to a29 at q_dropout=0.

Hypothesis (H19_query_dropout): a36 (no-diversity, val 0.800) BEAT a35
(orthogonality, val 0.775) by +0.025 -> the K=4 queries' learned overlap
is a useful feature the head exploits, and any *hard* diversity prior
breaks it. Per-query dropout is a stochastic regulariser that:
  (a) respects the overlap (no penalty term, queries free to correlate);
  (b) forces head robustness (must not depend on any one query);
  (c) acts only at training (zero inference-side risk).

If a37 lifts val above a29's 0.7997, the K=4 narrow peak from the
K-sweep (DE25) is explainable as "right capacity, wrong regularisation"
and we have a real direction. If a37 ties or loses, multi-query is
already saturated for this dataset and we close the family.

Mechanism (exact, train mode only):
    bag_per_query = attn @ h                         # [K, hidden]
    if training and q_dropout > 0:
        mask  = Bernoulli(1 - q_dropout)             # [K, 1]
        keep  = mask / (1 - q_dropout)               # inverse scaling
        bag_per_query = bag_per_query * keep
    bag = bag_per_query.reshape(-1)                  # [K * hidden]
    y   = clamp(Linear(K*hidden, 1)(bag), 0, 3)

Per-query mask, NOT per-element -> dropping a query zeros its entire
hidden-dim slice in the flatten, exactly the failure mode the head
must learn to absorb.

Ablation companion: `a38_query_dropout_off.py` (q_dropout=0; bit-for-
bit positive control for a29).

Kill criterion: abandon H19 if a37 val_qwk < a29 (0.7997) at seed=2.
If a37 ties a29 within +/- 0.005 val, declare multi-query saturated
and do NOT try further per-query regularisers.

Param count: identical to a29 = 172,993 (dropout adds no parameters).
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
        query_dim: int = 64,
        n_queries: int = 4,
        dropout: float = 0.5,
        clamp_output: bool = True,
        query_init_std: float = 0.02,
        q_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_queries >= 1
        assert 0.0 <= q_dropout < 1.0

        self.n_queries = n_queries
        self.scale = query_dim ** -0.5
        self.clamp_output = clamp_output
        self.q_dropout = q_dropout

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        self.queries = nn.Parameter(torch.randn(n_queries, query_dim) * query_init_std)
        self.classifier = nn.Linear(n_queries * hidden_dim, num_classes)
        nn.init.constant_(self.classifier.bias, 1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (single bag, batch_size=1 trainer contract).
        h = self.bottleneck(features)                       # [N, hidden]
        k = self.key_proj(h)                                # [N, q_dim]

        scores = self.queries @ k.t() * self.scale          # [K, N]
        attn = F.softmax(scores, dim=1)                     # [K, N]

        bag_per_query = attn @ h                            # [K, hidden]

        # Per-query stochastic dropout (train mode only). Drops an entire
        # query's hidden-dim slice in the subsequent flatten; survivors
        # are inverse-scaled so the expected magnitude is preserved.
        if self.training and self.q_dropout > 0.0:
            keep_prob = 1.0 - self.q_dropout
            # [K, 1] mask broadcast across hidden dim.
            mask = torch.empty(
                self.n_queries, 1, device=bag_per_query.device, dtype=bag_per_query.dtype
            ).bernoulli_(keep_prob).div_(keep_prob)
            bag_per_query = bag_per_query * mask

        bag = bag_per_query.reshape(-1)                     # [K * hidden]
        y = self.classifier(bag.unsqueeze(0)).squeeze(0)    # [1]

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attn.mean(dim=0), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    q_dropout=0.25,
)

