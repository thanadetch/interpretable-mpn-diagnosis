"""a423 - alpha = 1 + sigmoid(GAIN*(w1*s + w2*s^2 + b)), a curve that may peak inside the magnification range rather than running monotonically. a413 hinted at a step rather than a ramp. Control is a424."""
from .scale_formula_v2 import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False, form="quad", gain=10.0)
