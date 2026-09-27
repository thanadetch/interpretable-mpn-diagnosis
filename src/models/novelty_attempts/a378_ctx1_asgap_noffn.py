"""a378_ctx1_asgap_noffn — contextualised ASGAP variant.

a374 without the feed-forward sub-layer: attention-only contextualisation, fewer parameters.

    layers=1  readout='entmax'  pos='none'  ffn=False

Mechanism and rationale: see ``ctx_asgap.py``.
"""
from __future__ import annotations

from .ctx_asgap import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, layers=1, readout="entmax", pos="none", ffn=False)
