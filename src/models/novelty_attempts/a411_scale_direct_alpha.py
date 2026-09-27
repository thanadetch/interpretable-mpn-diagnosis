"""a411 — ONE learned entmax order per ROI, alpha = 1 + sigmoid(w_s*scale + w_u*unknown + b).

Main configuration; see `scale_direct_alpha.py`. Unlike the five-order mixture (a409) alpha here
moves freely over (1, 2), so this is the sharper test of whether magnification determines the
right sparsity. Control is a412 (same model, no scale input).
"""
from .scale_direct_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, condition=True)
