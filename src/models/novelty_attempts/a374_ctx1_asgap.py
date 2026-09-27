"""a374_ctx1_asgap — contextualised ASGAP variant.

ONE contextualisation block, then ASGAP pooling. The direct test of TransMIL's only
architectural difference from ABMIL/ASGAP, with pooling held fixed. Zero-initialised
residuals mean it starts bit-identical to ASGAP.

    layers=1  readout='entmax'  pos='none'  ffn=True

Mechanism and rationale: see ``ctx_asgap.py``.
"""
from __future__ import annotations

from .ctx_asgap import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, layers=1, readout="entmax", pos="none", ffn=True)
