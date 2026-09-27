"""a147 — ablation of a146: use_occupancy=False (drops the nonlinear selectivity-occupancy term).
= pure diffuse fibrosis-density readout (≈ a96). a146 vs a147 isolates EXACTLY whether a
genuinely-nonlinear, non-mean-reducible fibrosis-selective occupancy adds grade signal over
mean density — i.e. whether mean-sufficiency extends to nonlinear functionals. Reads data_distract.
"""
from __future__ import annotations
from .a146_selectivity_occupancy_density import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, use_occupancy=False, warm_start=True)
