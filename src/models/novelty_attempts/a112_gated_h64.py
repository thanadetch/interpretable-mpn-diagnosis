"""a112 — baseline gated-attention with hidden_dim=64 (capacity-width sweep).

See a111. a112 (H=64) / a111 (H=32) vs a104 (H=128 baseline) sweeps the
bottleneck width to locate the U-shape robustness optimum.
"""
from __future__ import annotations

from .a103_warmstart_gated import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=64, dropout=0.5, warm_start=False)
