"""a417 — alpha from a continuous formula of the measured um/px; unreadable scale bars pinned at 1.5.

The main configuration. Formula, initialisation and the reason the unknown group is pinned are
documented in `scale_formula_alpha.py`. Its control is a418, which permutes the scale assignment
while keeping the unknowns pinned in both arms; a417 is only interpretable next to it.
"""
from .scale_formula_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
