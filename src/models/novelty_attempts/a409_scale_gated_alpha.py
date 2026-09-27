"""a409 — learned gate over five entmax orders, conditioned on the ROI's measured um/px.

Main configuration. Mechanism, the 15-parameter gate and the evidence it has to overcome are in
`scale_gated_alpha.py`. Its control is a410, identical except the gate receives a constant input
and can therefore learn only ONE global mixture of the five orders. a409 is interpretable only
next to a410, because blending several sparsity levels is already known to lift G1 on its own.
"""
from .scale_gated_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, condition=True)
