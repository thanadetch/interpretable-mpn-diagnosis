"""a154 — ablation of a153: use_grading=False (grading bias OFF) = pure baseline ABMIL.
a153 vs a154 isolates whether ADDING a grading bias to the smart baseline's attention beats the
baseline. a154 should reproduce the baseline (sanity that the wrap pipeline is correct). Reads data_distract.
"""
from __future__ import annotations
from .a153_grading_wrapped_gated import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, use_grading=False)
