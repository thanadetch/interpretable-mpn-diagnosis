"""a428 — control for a427: the same four rows of four orders on a permuted band assignment.

The pool the bands are drawn from includes the unknown group, so the band marginal is preserved
exactly and only the ROI-to-band correspondence is destroyed.
"""
from .scale_pick_pinned import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, pin=False)
