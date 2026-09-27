"""a429 — two-stage alpha: each magnification band picks an order from {1.00, 1.25, 1.50, 2.00},
then that order becomes the starting point of a per-band learnable refinement.

The main configuration. Tests whether a215's inert learnable alpha was inert because 1.5 is a
poor starting point rather than because 30 patients cannot move a scalar. Control is a430.
Details in `scale_two_stage_alpha.py`.
"""
from .scale_two_stage_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
