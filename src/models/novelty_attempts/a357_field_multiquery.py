"""a357_field_multiquery — Field-Conditioned MIL variant.

MULTI-QUERY. The field view emits K=4 queries; each pools the bag independently and the K
descriptors are concatenated. One field can ask several questions (density, coarseness,
background) instead of one.

    attn='queryk'  pool='entmax'  readout='concat'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="queryk",
    pool="entmax",
    readout="concat",
    field="roi",
)
