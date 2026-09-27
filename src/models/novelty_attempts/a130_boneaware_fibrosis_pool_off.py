"""a130 — ablation of a129: bone_aware=False (pure diffuse fibrosis projection).

Isolates the bone-aware pooling: a129 vs a130 differ ONLY by whether bone-specific
patches are down-weighted in the diffuse pool. a130 = uniform diffuse mean -> fibrosis
projection -> affine (= the a96 diffuse-projection mechanism on the same features).
"""
from __future__ import annotations
from .a129_boneaware_fibrosis_pool import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, bone_aware=False, warm_start=True)
