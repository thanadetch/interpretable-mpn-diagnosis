"""a341_field_film — Field-Conditioned MIL variant.

FULL FC-MIL (FiLM variant). The field both re-scales the attention feature space and
contributes evidence. Compare against a340 (no conditioning) and a344 (no evidence path).

    attn='film'  pool='entmax'  readout='concat'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

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
    field="roi",
)
