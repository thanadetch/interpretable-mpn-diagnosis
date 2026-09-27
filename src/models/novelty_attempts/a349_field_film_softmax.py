"""a349_field_film_softmax — Field-Conditioned MIL variant.

a341 on top of dense softmax attention, i.e. FC-MIL applied to ABMIL rather than ASGAP.
Needed so the field mechanism can be credited independently of entmax sparsity.

    attn='film'  pool='softmax'  readout='concat'  field='roi'   base: ABMIL (softmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="film",
    pool="softmax",
    readout="concat",
    field="roi",
)
