"""a356_field_query_2step — Field-Conditioned MIL variant.

TWO-ROUND QUERY. Round 1 pools with the field query; the resulting descriptor then refines
the query for round 2 (coarse field -> first read -> corrected read). The refinement is
zero-initialised, so the model starts exactly as a342 and has to earn the second round.

    attn='query2'  pool='entmax'  readout='concat'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="query2",
    pool="entmax",
    readout="concat",
    field="roi",
)
