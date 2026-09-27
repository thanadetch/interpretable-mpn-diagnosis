"""a424 - control for a423: same quadratic form on a permuted scale assignment."""
from .scale_formula_v2 import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, form="quad", gain=10.0)
