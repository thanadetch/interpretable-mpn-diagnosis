"""a351_field_concat_softmax — Field-Conditioned MIL variant.

a340 on top of dense softmax attention (ABMIL base) — late fusion, no conditioning.

    attn='gated'  pool='softmax'  readout='concat'  field='roi'   base: ABMIL (softmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="gated",
    pool="softmax",
    readout="concat",
    field="roi",
)
