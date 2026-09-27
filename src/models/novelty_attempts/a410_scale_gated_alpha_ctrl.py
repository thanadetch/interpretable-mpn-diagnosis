"""a410 — CONTROL for a409: the same five-order gate, but it cannot see the scale.

The gate input is a constant, so a410 learns a single global mixture of alpha = 1.00..2.00 for
every ROI. Any advantage a409 holds over a410 is attributable to knowing the magnification;
anything they share belongs to the blending itself.
"""
from .scale_gated_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, condition=False)
