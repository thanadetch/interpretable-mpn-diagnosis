"""a343_field_gate — Field-Conditioned MIL variant.

ARBITRATION READOUT. A per-ROI gate decides how much of the prediction comes from the
patch pooling versus the field view directly. Tests whether the two magnifications are
better switched between than summed (a341).

    attn='film'  pool='entmax'  readout='gate'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="film",
    pool="entmax",
    readout="gate",
    field="roi",
)
