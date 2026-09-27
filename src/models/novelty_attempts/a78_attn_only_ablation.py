"""a78 — ablation of a77: gated attention ONLY (no explicit grading features).

use_grading_feats=False -> the head sees only the attention-weighted bag-rep,
i.e. exactly the baseline ABMIL aggregation. a77 vs a78 isolates
'do explicit diffuse-density distribution features (coverage/quantiles/spread
of the fibrosis projection) add over the learned gated-attention pooling?'.
"""
from __future__ import annotations

from .a77_attn_plus_grading_feats import Model  # noqa: F401  (re-exported)

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    use_grading_feats=False,
)
