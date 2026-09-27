"""a346_shuffled_field_film — Field-Conditioned MIL variant.

CONTROL for a341. q taken from a DIFFERENT ROI's field view. Same marginal distribution,
correspondence destroyed. If a341 does not beat this, the gain is not this ROI's field.

    attn='film'  pool='entmax'  readout='concat'  field='shuffle'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="film",
    pool="entmax",
    readout="concat",
    field="shuffle",
)
