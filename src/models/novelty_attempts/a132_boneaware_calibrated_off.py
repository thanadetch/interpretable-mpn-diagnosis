"""a132 — ablation of a131: bone_aware=False = calibrated diffuse fibrosis projection
(should reproduce ~a96). Isolates bone-avoidance under matched calibration.
"""
from __future__ import annotations
from .a131_boneaware_calibrated import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, bone_aware=False, warm_start=True)
