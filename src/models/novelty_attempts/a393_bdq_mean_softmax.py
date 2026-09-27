"""a393_bdq_mean_softmax — Bulk-Deflated Query MIL variant.

Pooling ablation — the same deflated scoring over dense softmax. Separates the deflation
mechanism from the entmax choice.

    eps=1.0  deflate='mean'  pool='softmax'

Mechanism, measured motivation and risks: see ``bdq_mil.py``.
"""
from __future__ import annotations

from .bdq_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, eps=1.0, deflate="mean", pool="softmax")
