"""a376_ctx1_cls — contextualised ASGAP variant.

ONE contextualisation block with a CLASS-TOKEN readout instead of ASGAP pooling. Isolates
the readout: same contextualisation as a374, TransMIL's aggregation.

    layers=1  readout='cls'  pos='none'  ffn=True

Mechanism and rationale: see ``ctx_asgap.py``.
"""
from __future__ import annotations

from .ctx_asgap import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, layers=1, readout="cls", pos="none", ffn=True)
