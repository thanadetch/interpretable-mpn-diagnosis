"""a401_tsa_shared_attn — Threshold-Specific Attention MIL variant.

THE DECIDING CONTROL. Identical in every respect except that ONE attention map is shared
by all three thresholds. If a400 does not beat this, the gain is the ordinal chain, not
threshold-specific attention, and the mechanism claim is dead.

    thresholds=3  share_attn=True  monotone=True  pool='entmax'

Mechanism, motivation and the falsification plan: see ``tsa_mil.py``.
"""
from __future__ import annotations

from .tsa_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, thresholds=3, share_attn=True,
              monotone=True, pool="entmax")
