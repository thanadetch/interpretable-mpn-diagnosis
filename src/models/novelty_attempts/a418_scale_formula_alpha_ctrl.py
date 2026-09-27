"""a418 — control for a417: the same formula on a PERMUTED scale assignment.

Identical parameter count and identical alpha marginal; only the correspondence between an ROI
and its own measured um/px is destroyed. The unknown group stays pinned at 1.5 in both arms, so
the pair differs in exactly one thing. If a418 matches a417, the measurement carries nothing.
"""
from .scale_formula_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True)
