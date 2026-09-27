"""a359_field_resid_query_softmax — Field-Conditioned MIL variant.

a358 on a dense-softmax (ABMIL) base.

    attn='query'  pool='softmax'  readout='concat'  field='resid'   base: ABMIL (softmax pooling)

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
    field="resid",
)
