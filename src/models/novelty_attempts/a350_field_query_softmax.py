"""a350_field_query_softmax — Field-Conditioned MIL variant.

a342 on top of dense softmax attention (ABMIL base).

    attn='query'  pool='softmax'  readout='concat'  field='roi'   base: ABMIL (softmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="query",
    pool="softmax",
    readout="concat",
    field="roi",
)
