"""a345_meanfield_film — Field-Conditioned MIL variant.

CONTROL for a341. q = bottleneck(mean patch): identical architecture, identical parameter
count, identical per-bag adaptivity, but NO information the bag did not already contain.
If a341 does not beat this, the gain is adaptivity, not the un-patched field view.

    attn='film'  pool='entmax'  readout='concat'  field='mean'   base: ASGAP (alpha=1.5 entmax pooling)

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
    field="mean",
)
