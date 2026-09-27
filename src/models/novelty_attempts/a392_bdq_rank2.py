"""a392_bdq_rank2 — Bulk-Deflated Query MIL variant.

Deflates the mean direction AND the top principal direction of the centered bag —
is one nuisance direction enough, or two?

    eps=1.0  deflate='rank2'  pool='entmax'

Mechanism, measured motivation and risks: see ``bdq_mil.py``.
"""
from __future__ import annotations

from .bdq_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, eps=1.0, deflate="rank2", pool="entmax")
