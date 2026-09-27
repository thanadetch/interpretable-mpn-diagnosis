"""a390_bdq_mean — Bulk-Deflated Query MIL variant.

PRIMARY. Full deflation of the bag's bulk (mean) direction from the attention scores,
entmax pooling. The mechanism claim lives or dies on this row beating a394 (eps=0).

    eps=1.0  deflate='mean'  pool='entmax'

Mechanism, measured motivation and risks: see ``bdq_mil.py``.
"""
from __future__ import annotations

from .bdq_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, eps=1.0, deflate="mean", pool="entmax")
