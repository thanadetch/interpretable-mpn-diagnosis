"""a144 — ablation of a143: norm_mode="global" (global concept-score constants).
= the a135 bone-aware diffuse-density mechanism. a143 (per_bag) vs a144 (global) isolates
EXACTLY the effect of per-ROI vs global bone/fib normalization. Reads data_bonefib.
"""
from __future__ import annotations
from .a143_perbag_boneaware_density import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, norm_mode="global", warm_start=True)
