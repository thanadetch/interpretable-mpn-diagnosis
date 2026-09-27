"""a404_tsa_k1 — Threshold-Specific Attention MIL variant.

SANITY FLOOR. K=1 collapses to a single attention with a sigmoid readout - the baseline
shape, no ordinal decomposition at all.

    thresholds=1  share_attn=False  monotone=True  pool='entmax'

Mechanism, motivation and the falsification plan: see ``tsa_mil.py``.
"""
from __future__ import annotations

from .tsa_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, thresholds=1, share_attn=False,
              monotone=True, pool="entmax")
