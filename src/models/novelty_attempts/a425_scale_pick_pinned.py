"""a425 — each measured magnification band picks one of {1.00, 1.25, 1.50, 2.00}; ROIs whose
scale bar cannot be read are pinned at 1.5 with no gradient path.

The main configuration. This is a415 re-run under a417's pinning, which is the change that first
made a scale model separate from its control. Its control is a426; a425 is only interpretable
next to it. Details in `scale_pick_pinned.py`.
"""
from .scale_pick_pinned import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
