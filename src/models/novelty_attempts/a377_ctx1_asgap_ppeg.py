"""a377_ctx1_asgap_ppeg — contextualised ASGAP variant.

a374 plus TransMIL's PPEG position block. Costs permutation invariance for an ARBITRARY
token grid; this cohort has already been measured to carry no signal in the REAL patch grid,
so a loss here is the expected outcome and a direct ablation of TransMIL's position claim.

    layers=1  readout='entmax'  pos='ppeg'  ffn=True

Mechanism and rationale: see ``ctx_asgap.py``.
"""
from __future__ import annotations

from .ctx_asgap import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, layers=1, readout="entmax", pos="ppeg", ffn=True)
