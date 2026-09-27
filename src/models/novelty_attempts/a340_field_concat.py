"""a340_field_concat — Field-Conditioned MIL variant.

FUSION ONLY. The field vector reaches the prediction head but never touches the attention.
Ablates FC-MIL down to late fusion of two magnifications, so a341/a342 have to earn the
conditioning machinery rather than just the extra evidence.

    attn='gated'  pool='entmax'  readout='concat'  field='roi'   base: ASGAP (alpha=1.5 entmax pooling)

Mechanism, controls and rationale: see ``field_mil.py``.
"""
from __future__ import annotations

from .field_mil import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    attn="gated",
    pool="entmax",
    readout="concat",
    field="roi",
)
