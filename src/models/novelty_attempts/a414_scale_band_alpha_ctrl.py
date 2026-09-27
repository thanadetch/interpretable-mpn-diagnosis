"""a414 — CONTROL for a413: four free learnable orders, but bands assigned to ROIs at random.

Same capacity, same band sizes, correspondence to the real magnification destroyed. If a414
matches a413, the grouping carries no information and only the freedom to use several orders
matters.
"""
from .scale_band_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, seed=0)
