"""a348_shuffled_field_query — Field-Conditioned MIL variant.

CONTROL for a342 — mismatched field query.

    attn='query'  pool='entmax'  readout='concat'  field='shuffle'   base: ASGAP (alpha=1.5 entmax pooling)

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
    field="shuffle",
)
