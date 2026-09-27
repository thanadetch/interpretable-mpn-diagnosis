"""a360_static_query — Field-Conditioned MIL control.

DECISIVE CONTROL 1 — no conditioning at all. A single LEARNED query, identical for every
bag, with the same dot-product attention as a342. If this already beats gated-attention
ASGAP, the gain is 'dot-product attention instead of gated attention' and has nothing to do
with the field view or with per-bag conditioning.

    attn='query'  pool='entmax'  readout='concat'  field='static'   base: ASGAP (alpha=1.5 entmax)

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
    field="static",
)
