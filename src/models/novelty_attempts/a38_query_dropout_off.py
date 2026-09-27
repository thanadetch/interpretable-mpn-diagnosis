"""a38 - K=4 multi-query cross-attn, query-dropout DISABLED (H19, batch 19 ablation).

Bit-for-bit positive control for a29 / a36. Reuses a37.Model with
q_dropout=0.0 so the train-mode mask branch is never entered.

Active ingredient being isolated (vs a37): the train-time per-query
dropout mask.

Interpretation matrix:
    a37 val > a38 val   -> query-dropout helps; H19 succeeds.
    a37 val ~= a38 val  -> query-dropout neutral; close the family.
    a37 val < a38 val   -> query-dropout HURTS; head needs all K
                            queries always-on (same pattern as DE26).

Kill criterion: shared with a37 (family-level).

Param count: 172,993 (identical to a29 / a36).
"""
from __future__ import annotations

from .a37_query_dropout import Model as _Model

Model = _Model

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    q_dropout=0.0,
)

