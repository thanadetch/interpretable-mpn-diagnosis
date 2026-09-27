"""a361_random_query — Field-Conditioned MIL control.

DECISIVE CONTROL 2 — per-bag query carrying zero information. A fixed pseudo-random vector
seeded by the bag's own fingerprint, scale-matched to the bag mean, never touching the field
bank. If this matches a342, the field bank is irrelevant and the effect is 'any per-bag
query that is not the bag centroid'.

    attn='query'  pool='entmax'  readout='concat'  field='random'   base: ASGAP (alpha=1.5 entmax)

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
    field="random",
)
