"""a412 — CONTROL for a411: one global learnable alpha, no scale input.

alpha = 1 + sigmoid(b), shared by every ROI. Isolates the scale input: whatever a411 gains over
this row is what knowing the magnification bought.
"""
from .scale_direct_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, condition=False)
