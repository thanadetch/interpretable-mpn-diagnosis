"""a422 - control for a421: same clamped-linear form on a permuted scale assignment."""
from .scale_formula_v2 import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, form="linear", gain=1.0)
