"""a391_bdq_mean_half — Bulk-Deflated Query MIL variant.

Partial deflation (eps=0.5) — hedges the G3 risk: in a uniformly fibrotic bag the grade
signal may itself be the bulk direction.

    eps=0.5  deflate='mean'  pool='entmax'

Mechanism, measured motivation and risks: see ``bdq_mil.py``.
"""
from __future__ import annotations

from .bdq_mil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, eps=0.5, deflate="mean", pool="entmax")
