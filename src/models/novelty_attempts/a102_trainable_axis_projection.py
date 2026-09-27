"""a102 — ablation of a101: TRAINABLE axis (zero-DOF freeze removed).

Imports a101's Model with train_axis=True so the projection direction is a
warm-started Parameter that can drift to fit the fold. a101 (frozen) vs a102
(trainable) isolates whether removing all direction-DOF reduces the val<->test
gap on this small cohort.
"""
from __future__ import annotations

from .a101_frozen_axis_projection import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, train_axis=True)
