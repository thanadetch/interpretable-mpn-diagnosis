"""a106 — ablation of a105: n_points=1 (sorted curve collapsed to central density).

a105 (M=16 sorted curve) vs a106 (M=1 = mean of the projection) isolates whether
the full sorted distribution SHAPE adds over the central diffuse density.
"""
from __future__ import annotations

from .a105_sorted_curve_pool import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_points=1)
