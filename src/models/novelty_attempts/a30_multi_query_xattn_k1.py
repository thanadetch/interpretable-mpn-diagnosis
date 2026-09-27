"""a30 - single-query cross-attention K=1 (H15 ablation companion to a29).

Identical architecture to `a29_multi_query_xattn.py` but with K=1
(a single learnable query). This isolates whether *having multiple*
queries is the active ingredient, or whether the dot-product cross-
attention architecture itself (irrespective of K) is what helps or
hurts.

Expected outcomes (vs a29 K=4):
    - a29 val > a30 val by >= 0.005    -> multi-query specialisation IS the
                                          active ingredient; H15 succeeds.
    - a29 val ~= a30 val                -> the K queries collapse to redundant
                                          views; architecture not multiplicity
                                          carries any gain.
    - a29 val < a30 val                 -> more queries overfit; revert to K=1.

Kill criterion: shared with a29 (family-level). Abandon H15 if BOTH
a29 and a30 val_qwk < a17's 0.7970 at seed=2.

Param count (K=1):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=1, q_dim=64]                 =      64
    classifier Linear(128, 1) + bias           =     129
    --------------------------------------------------
    total trainable                            = 172,417
    (vs ABMIL 197,250 -> -24,833 params)
"""
from .a29_multi_query_xattn import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=1,            # <-- the only flipped flag vs a29
    dropout=0.5,
    query_init_std=0.02,
)

