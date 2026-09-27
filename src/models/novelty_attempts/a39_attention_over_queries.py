"""a39 - learned soft-attention over the K=4 query bag-reps (H20 main, batch 20).

Hint targeted (from NOVELTY_NOTES.md §7):
    H20_attention_over_queries - replace the flatten + Linear(K*hidden, 1) head
    of a29 with a learned soft-attention over the K=4 query bag-reps that
    fuses them into a single [hidden] vector before a Linear(hidden, 1).

Hypothesis: batches 16-19 confirmed the K=4 multi-query base (a29, val
0.7997) is a *narrow* local optimum that resists every perturbation tested
so far - stacking the H10 mechanisms (DE24), sweeping K (DE25), forcing
query orthogonality (DE26), and stochastic query dropout (DE27) all left
val at or below 0.800. The one architectural piece we have NOT touched is
the head: a29 flattens the K bag-reps and feeds K*hidden=512 features to
a single Linear(512, 1). That head treats the K queries as a fixed
ordered vector - the head's weights *implicitly* learn per-query gating,
but only through one linear map jointly with the per-query feature
mixing. Replacing it with an *explicit* learned soft-attention over the K
bag-reps decouples the two roles: the attn scorer learns which queries
to trust per bag, then a smaller Linear(hidden, 1) maps the fused
[hidden] vector to ŷ. The expectation is that bag-conditional query
selection should help the val cohort where the K=4 base is currently
giving every query equal architectural standing.

Mechanism (exact):
    h_i        = bottleneck(features_i)                # [N, hidden=128]
    k_i        = Linear(h_i)                           # [N, q_dim=64]
    Q          = learnable parameters                  # [K=4, q_dim=64]
    scores     = Q @ k.T * q_dim**-0.5                 # [K, N]
    attn_patch = softmax(scores, dim=N)                # [K, N]
    bag_per_q  = attn_patch @ h                        # [K, hidden]
    # === Head pivot vs a29 ===
    s          = Linear(hidden, 1)(bag_per_q)          # [K, 1] - per-query scalar score
    attn_q     = softmax(s, dim=K)                     # [K, 1] - one weight per query
    bag        = (attn_q * bag_per_q).sum(dim=0)       # [hidden] - fused bag rep
    y          = clamp(Linear(hidden, 1)(bag), 0, 3)

No sigmoid between bag rep and y (avoids DE11-13 gradient bottleneck).
No norm-based per-patch prior (avoids DE15 ceiling). Attention is over
the K=4 queries, not over N patches, so it cannot overfit per-patch noise
the way per-patch attention re-weights can.

Ablation companion: `a40_mean_over_queries.py` - identical architecture
but the K bag-reps are averaged with uniform 1/K weights (no learned
attn_q scorer). Tells us cleanly whether *learned per-bag query gating*
is the active ingredient vs simply "fuse instead of flatten".

Kill criterion: abandon H20 if a39 val_qwk < a29's 0.7997 at seed=2
(i.e., explicit query attention does not beat the implicit-via-flatten
head a29 already has). Pre-registered before running.

Param count (input_dim=1280, hidden_dim=128, q_dim=64, K=4):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=4, q_dim=64]                 =     256
    attn_scorer Linear(128, 1) + bias          =     129
    classifier  Linear(128, 1) + bias          =     129
    --------------------------------------------------
    total trainable                            = 172,738
    (vs a29 172,993 -> -255 params; vs baseline 197,250 -> -24,512)
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
        learned_query_attention: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_queries >= 1

        self.n_queries = n_queries
        self.scale = query_dim ** -0.5
        self.clamp_output = clamp_output
        self.learned_query_attention = learned_query_attention

        # Same bottleneck shape as a29 / ABMIL.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        self.queries = nn.Parameter(torch.randn(n_queries, query_dim) * query_init_std)

        # Per-query attention scorer (the H20 pivot).
        # When learned_query_attention=False (ablation a40), the scorer is
        # never used and the fusion is uniform 1/K mean.
        if self.learned_query_attention:
            self.query_attn = nn.Linear(hidden_dim, 1)

        # Smaller head: Linear(hidden, 1) on a fused [hidden] vector.
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # Bias-init to prior mean to avoid dead-clamp at random init.
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

        # Cross-attention over patches.
        scores = self.queries @ k.t() * self.scale          # [K, N]
        attn_patch = F.softmax(scores, dim=1)               # [K, N]
        bag_per_query = attn_patch @ h                      # [K, hidden]

        # === H20 pivot: fuse K bag-reps via learned soft-attention. ===
        if self.learned_query_attention:
            s = self.query_attn(bag_per_query)              # [K, 1]
            attn_q = F.softmax(s, dim=0)                    # [K, 1]
        else:
            attn_q = bag_per_query.new_full(
                (self.n_queries, 1), 1.0 / self.n_queries
            )                                                # [K, 1] uniform
        bag = (attn_q * bag_per_query).sum(dim=0)           # [hidden]

        y = self.classifier(bag.unsqueeze(0)).squeeze(0)    # [1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            # Return the per-patch attention averaged over queries
            # (interpretability slot expected by the trainer).
            return y, attn_patch.mean(dim=0), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    learned_query_attention=True,   # a39 main: learned soft-attention over queries
)

