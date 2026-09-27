"""a40 - mean over the K=4 query bag-reps (H20 ablation companion, batch 20).

Ablation companion to `a39_attention_over_queries.py`. Identical
architecture except the K bag-reps are fused with **uniform 1/K weights**
instead of a learned soft-attention. Isolates "learned per-bag query
gating" as the active ingredient of a39.

Compared to a29 (which flattens the K bag-reps and runs a single
Linear(K*hidden, 1)), a40 still pivots the head to a fused [hidden] -> 1
mapping, just without per-query weighting. So the a39/a40 contrast is
clean ("learned vs uniform fusion"), and the a39/a40 vs a29 contrast
covers the orthogonal axis ("fuse vs flatten").

Kill criterion: this is the ablation - it is expected to underperform a39
or at most match it. No independent kill threshold beyond the H20 family
threshold in a39's docstring.

Param count (input_dim=1280, hidden_dim=128, q_dim=64, K=4):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=4, q_dim=64]                 =     256
    classifier  Linear(128, 1) + bias          =     129
    --------------------------------------------------
    total trainable                            = 172,609
    (vs a39 172,738 -> -129 params, removes the query_attn scorer)
"""
from __future__ import annotations

from .a39_attention_over_queries import Model as _Base


class Model(_Base):
    """Inherits a39's architecture; switches off the learned query attention."""


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    learned_query_attention=False,  # ablation: uniform 1/K mean over queries
)

