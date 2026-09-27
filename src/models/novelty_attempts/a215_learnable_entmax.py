"""DEPRECATED ALIAS — kept only for backward compatibility.

The model was renamed to ``a215_adaptive_sparse_gated_attention_pooling`` (ASGAP — Adaptive-Sparse Gated Attention
Pooling). The old id ``a215_learnable_entmax`` is recorded in ~43 historical
experiment configs (``experiments/a215_ablation/`` and the date-bucketed dirs);
this alias keeps those runs reproducible via ``--novelty_id a215_learnable_entmax``.

New code should import / use ``a215_adaptive_sparse_gated_attention_pooling``. ("learnable" was a misnomer — the
alpha is empirically inert, so it is effectively fixed adaptive-sparse pooling.)
"""
from .a215_adaptive_sparse_gated_attention_pooling import *  # noqa: F401,F403
from .a215_adaptive_sparse_gated_attention_pooling import Model, KWARGS, entmax_bisect  # noqa: F401
