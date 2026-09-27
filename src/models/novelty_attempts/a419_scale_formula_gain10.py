"""a419 - a417's form with GAIN=10 instead of 3. The decisive variant: if the alpha span grows with GAIN the earlier runs were limited by the parameterisation; if w_s shrinks to keep the span fixed, the loss was already at its optimum. Control is a420. Details in scale_formula_v2.py."""
from .scale_formula_v2 import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False, form="gain", gain=10.0)
