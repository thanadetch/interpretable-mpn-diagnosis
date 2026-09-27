"""a394_bdq_off — Bulk-Deflated Query MIL variant.

THE CONTROL. eps=0: identical architecture, no deflation — a pure learned static query
with entmax. Any a390 gain that this row matches is not the deflation.

    eps=0.0  deflate='mean'  pool='entmax'

Mechanism, measured motivation and risks: see ``bdq_mil.py``.
"""
from __future__ import annotations

from .bdq_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, eps=0.0, deflate="mean", pool="entmax")
