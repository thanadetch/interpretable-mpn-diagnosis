"""a426 — control for a425: the same three rows of four orders, but each measurable ROI is given a
band drawn from the observed band distribution rather than its own. Identical parameter count and
identical alpha marginal; only the ROI-to-magnification correspondence is destroyed. The
unmeasurable ROIs stay pinned at 1.5 in both arms.
"""
from .scale_pick_pinned import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True)
