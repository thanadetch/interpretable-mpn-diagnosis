"""a142 — ablation of a141: use_uniformity=False (drops the dispersion term).
= pure diffuse fibrosis-density readout (≈ a96). a141 vs a142 isolates EXACTLY whether
within-bag meshwork uniformity (dispersion of the per-patch fibrosis score) adds grade
signal over fibre amount. Reads plain `data` (D-dim).
"""
from __future__ import annotations
from .a141_uniformity_diffuse_density import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, use_uniformity=False, warm_start=True)
