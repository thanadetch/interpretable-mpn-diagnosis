"""a403_tsa_softmax — Threshold-Specific Attention MIL variant.

POOLING ABLATION. a400 over dense softmax instead of entmax, so the mechanism can be
credited independently of sparse pooling.

    thresholds=3  share_attn=False  monotone=True  pool='softmax'

Mechanism, motivation and the falsification plan: see ``tsa_mil.py``.
"""
from __future__ import annotations

from .tsa_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, thresholds=3, share_attn=False,
              monotone=True, pool="softmax")
