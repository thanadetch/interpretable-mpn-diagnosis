"""a430 — control for a429: both stages run identically on a permuted band assignment, so the
band marginal is preserved and only the ROI-to-magnification correspondence is destroyed.
"""
from .scale_two_stage_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True)
