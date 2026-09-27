"""a34 - multi-query cross-attention with K=2 (H16 ablation companion to a33).

Together with a30 (K=1), a29 (K=4) and a33 (K=8), provides a clean
K-sweep: K in {1, 2, 4, 8}. Tests whether even minimal multiplicity
(K=2) lifts over K=1, and bounds the slope below K=4.

Kill criterion: shared with a33 (family-level). Abandon H16 if
both K=2 and K=8 fail to clarify the K-axis trend.

Param count (K=2):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=2, q_dim=64]                 =     128
    classifier Linear(K*hidden=256, 1) + bias  =     257
    --------------------------------------------------
    total trainable                            = 172,609
"""
from .a29_multi_query_xattn import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=2,            # <-- minimal multiplicity
    dropout=0.5,
    query_init_std=0.02,
)

