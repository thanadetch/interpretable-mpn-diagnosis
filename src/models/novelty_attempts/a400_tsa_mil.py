"""a400_tsa_mil — Threshold-Specific Attention MIL variant.

PRIMARY. Three ordinal thresholds, each with its OWN gated attention over the shared
instance embeddings, combined by a monotone chain. The claim: the evidence for >=MF-1
(any fibre at all) lives in different patches from the evidence for >=MF-3 (coarse
intersecting bundles), so one attention map cannot serve all three.

    thresholds=3  share_attn=False  monotone=True  pool='entmax'

Mechanism, motivation and the falsification plan: see ``tsa_mil.py``.
"""
from __future__ import annotations

from .tsa_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, thresholds=3, share_attn=False,
              monotone=True, pool="entmax")
