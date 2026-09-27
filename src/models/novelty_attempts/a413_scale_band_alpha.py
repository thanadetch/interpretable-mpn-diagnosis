"""a413 — one FREE learnable entmax order per magnification band (4 params, all start at 1.5).

The direct answer to "learn what alpha each scale wants": the bands are independent, so each may
settle anywhere in (1, 2) in any order. See `scale_band_alpha.py`. Control is a414.
"""
from .scale_band_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
