"""a353_ownpat_query — Field-Conditioned MIL variant.

OWN FIELD + PATIENT CONTEXT. q = W_q(own field) + W_c(leave-one-out patient context).
Tests whether the case-level context adds anything on top of the winning a342.

    attn='query'  pool='entmax'  readout='concat'  field='ownpat'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="query",
    pool="entmax",
    readout="concat",
    field="ownpat",
)
