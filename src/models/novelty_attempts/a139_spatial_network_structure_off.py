"""a139 — ablation of a138: use_structure=False (drops the Moran's-I spatial term).
= pure diffuse fibrosis-density readout (≈ a96). a138 vs a139 isolates EXACTLY whether
spatial network structure adds grade signal over fibre amount. Reads data_struct (1282-d);
the trailing rc cols are simply ignored.
"""
from __future__ import annotations
from .a138_spatial_network_structure import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, use_structure=False, warm_start=True)
