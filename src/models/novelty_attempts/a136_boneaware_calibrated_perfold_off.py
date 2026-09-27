"""a136 — ablation of a135: bone_aware=False = per-fold calibrated DIFFUSE projection
(~ a96 with per-fold warm axis + per-fold anchor calibration). Isolates bone-avoidance
under matched per-fold calibration, for the 5-fold MEAN comparison.
"""
from __future__ import annotations
from .a135_boneaware_calibrated_perfold import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, bone_aware=False, warm_start=True)
