"""a33 - multi-query cross-attention with K=8 (H16_multi_query_K_sweep, batch 17 main).

Tests whether the K=4 > K=1 ordering from batch 15 (a29 0.7997 > a30
0.7822) continues monotonically when we double K. If K=8 > K=4, we
have a real "more queries -> better val" direction. If K=8 < K=4,
K=4 was a noisy local peak. If K=8 ~ K=4, the gain saturates at K=4.

Pathology rationale: 8 concept queries can each specialise on a
different reticulin sub-pattern (fine fibres, coarse bundles,
replacement, nodules, vessels, background, etc.).

Ablation companion: `a34_multi_query_xattn_k2.py` (K=2). Together
with a30 (K=1) and a29 (K=4), gives a 4-point K-sweep at K in {1, 2, 4, 8}.

Kill criterion: abandon H16 if K=8 val < K=4 val (a29 = 0.7997).
If K=8 saturates within +/- 0.005, do NOT push to K=16.

Param count (input_dim=1280, hidden_dim=128, q_dim=64, K=8):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=8, q_dim=64]                 =     512
    classifier Linear(K*hidden=1024, 1) + bias =   1,025
    --------------------------------------------------
    total trainable                            = 173,761
"""
from .a29_multi_query_xattn import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=8,            # <-- swept axis vs a29 (K=4) and a30 (K=1)
    dropout=0.5,
    query_init_std=0.02,
)

