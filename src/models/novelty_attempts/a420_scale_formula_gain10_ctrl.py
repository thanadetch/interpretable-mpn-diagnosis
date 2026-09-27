"""a420 - control for a419: identical form and GAIN, permuted scale assignment, unknowns pinned in both arms."""
from .scale_formula_v2 import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, form="gain", gain=10.0)
