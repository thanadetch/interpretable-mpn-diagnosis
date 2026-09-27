"""a342_field_query — Field-Conditioned MIL variant.

FIELD-AS-QUERY. ABMIL's attention query is one parameter shared by every bag; here it is
computed from the ROI's own field view. The sharpest form of the idea, and the only
variant with no fieldless limit.

    attn='query'  pool='entmax'  readout='concat'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

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
    field="roi",
)
