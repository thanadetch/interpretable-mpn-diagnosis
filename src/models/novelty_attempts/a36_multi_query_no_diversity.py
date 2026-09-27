"""a36 - K=4 multi-query cross-attn with lambda_div=0 (H18 ablation companion to a35).

Identical to a35 with the diversity hook disabled. This is bit-for-bit
the same as a29 (the H15 batch-15 module) up to the classifier-bias
init of 1.5. Acts as a positive control: should reproduce a29 val_qwk
exactly (0.7997).

Kill criterion: shared with a35 (family-level). Abandon H18 if a35 val
does not exceed a36 val by at least 0.005.
"""
from .a35_multi_query_diversity import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    lambda_div=0.0,
)
