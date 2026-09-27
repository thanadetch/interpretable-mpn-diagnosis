"""a32 - multi-query cross-attention + length-norm only (H17 ablation companion to a31).

Identical architecture to `a31_multi_query_coverage_lengthnorm.py` but
`use_coverage = False`. Length-normalised softmax temperature is kept;
the coverage prior (per-patch sigmoid of ||h_i||) is removed.

Active ingredient being isolated: the **coverage prior** on top of
multi-query + length-norm. Interpretation matrix:

    a31 val > a32 val   -> coverage IS additive on top of multi-query;
                            H10 stack composes with H15; H17 succeeds.
    a31 val ~= a32 val  -> coverage redundant inside multi-query; the
                            K queries already cover what the coverage
                            prior provided in single-head attention.
    a31 val < a32 val   -> coverage HURTS inside multi-query;
                            diagnostic Hint 3 (norm-rank saturated) is
                            right and the per-norm prior actually
                            interferes with the queries' specialisation.

Kill criterion: shared with a31 (family-level). Abandon H17 if BOTH
a31 and a32 val_qwk < a29 (0.7997) at seed=2.

Param count (use_coverage=False, K=4):
    bottleneck Linear(1280, 128) + bias       = 163,968
    key_proj   Linear(128,  64) + bias         =   8,256
    queries    [K=4, q_dim=64]                 =     256
    classifier Linear(K*hidden=512, 1) + bias  =     513
    c0_raw (length-norm temperature)           =       1
    --------------------------------------------------
    total trainable                            = 172,994
    (vs a31 172,997 -> -3 params; vs a29 172,993 -> +1 c0_raw)
"""
from .a31_multi_query_coverage_lengthnorm import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    use_coverage=False,     # <-- the only flipped flag vs a31
    tau_init=1.0,
    beta_init=1.0,
    alpha_init=1e-3,
    c0_init=6.324555,
)

