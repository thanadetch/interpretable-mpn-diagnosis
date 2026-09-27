"""a375_ctx2_asgap — contextualised ASGAP variant.

TWO contextualisation blocks (TransMIL's depth), ASGAP pooling.

    layers=2  readout='entmax'  pos='none'  ffn=True

Mechanism and rationale: see ``ctx_asgap.py``.
"""
from __future__ import annotations

from .ctx_asgap import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, layers=2, readout="entmax", pos="none", ffn=True)
