"""a29 - multi-query cross-attention with K=4 concept queries (H15, batch 15 main).

Hypothesis (H15_multi_query_xattn): batches 1-14 confirmed that every
*augmentation* on top of ABMIL's softmax-mean overfits (DE11-23)
and that *replacing* the softmax-mean with a frozen scalar scorer
under-fits (DE22). The remaining architectural axis we have NOT touched is
**learned multi-query cross-attention** (Perceiver / Set Transformer
PMA family): K learnable "concept queries" cross-attend to the bottleneck
features, producing K bag representations, which the head then combines.

Pathology rationale: reticulin fibrosis presents as several distinct
visual patterns (fine fibres, coarse bundles, replacement of stromal
architecture, fibrous nodules). A single attention head (a17) must
collapse all of these into one ranking; K queries can each specialise
on a different sub-pattern and let the head fuse them. The K queries
also act as a *structural bottleneck* (K << N): they cannot overfit
to per-patch noise the way per-patch attention weights can.

Mechanism (exact):
    h_i       = bottleneck(features_i)               # [N, hidden=128]
    k_i       = Linear(h_i)                          # [N, q_dim=64]
    Q         = learnable parameters                 # [K, q_dim=64]
    scores    = Q @ k.T * (q_dim**-0.5)              # [K, N]
    attn      = softmax(scores, dim=N)               # [K, N]; each query softmaxes over patches
    bag_q     = attn @ h                             # [K, hidden] -- one rep per query
    bag       = bag_q.flatten()                      # [K * hidden]
    y         = clamp(Linear(K*hidden, 1)(bag), 0, 3)

No sigmoid between bag rep and y -> no DE11-13 gradient bottleneck.
Values reuse the bottleneck output `h` (no separate V projection) to
keep params at ~173k, below the baseline's 197k.

Ablation companion: `a30_multi_query_xattn_k1.py` - identical
architecture but K=1 (single query). Tells us cleanly whether *having
multiple* queries is the active ingredient, or whether the cross-
attention architecture itself (irrespective of K) is what matters.

Kill criterion: abandon H15 if BOTH a29 and a30 val_qwk < a17's 0.7970
at seed=2.

Param count (input_dim=1280, hidden_dim=128, q_dim=64, K=4):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=4, q_dim=64]                 =     256
    classifier Linear(K*hidden=512, 1) + bias  =     513
    --------------------------------------------------
    total trainable                            = 172,993
    (vs ABMIL 197,250 -> -24,257 params, -12%)
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
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_queries >= 1

        self.n_queries = n_queries
        self.scale = query_dim ** -0.5
        self.clamp_output = clamp_output

        # Same bottleneck shape as ABMIL / a17.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Project bottleneck features to attention keys.
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        # Learnable concept queries.
        self.queries = nn.Parameter(torch.randn(n_queries, query_dim) * query_init_std)
        # Head: flatten K bag-reps then linear.
        self.classifier = nn.Linear(n_queries * hidden_dim, num_classes)
        # Bias-init to prior mean (1.5 = midpoint of [0, 3]) so the clamp at
        # forward end doesn't dead-zero gradients at random init when the
        # K*hidden flatten makes the pre-clamp output's variance large.
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

        # Cross-attention: [K, N] = [K, q_dim] @ [q_dim, N]
        scores = self.queries @ k.t() * self.scale          # [K, N]
        attn = F.softmax(scores, dim=1)                     # [K, N], rows sum to 1

        # K bag representations, one per query.
        bag_per_query = attn @ h                            # [K, hidden]
        bag = bag_per_query.reshape(-1)                     # [K * hidden]
        y = self.classifier(bag.unsqueeze(0)).squeeze(0)    # [1]

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Return mean attention across queries for interpretability.
            return y, attn.mean(dim=0), None                # [N]
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,            # a29 main: 4 concept queries
    dropout=0.5,
    query_init_std=0.02,
)


