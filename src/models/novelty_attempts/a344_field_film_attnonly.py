"""a344_field_film_attnonly — Field-Conditioned MIL variant.

CONDITIONING ONLY. FiLM on the attention, but the field never reaches the head. Isolates
whether steering WHERE the model looks is worth anything on its own.

    attn='film'  pool='entmax'  readout='none'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="film",
    pool="entmax",
    readout="none",
    field="roi",
)
