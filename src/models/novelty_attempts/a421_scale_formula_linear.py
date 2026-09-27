"""a421 - alpha = clamp(1.5 + w_s*s + b, 1, 2). No saturating envelope, so nothing bounds the slope except the ends of the valid range. Control is a422."""
from .scale_formula_v2 import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False, form="linear", gain=1.0)
