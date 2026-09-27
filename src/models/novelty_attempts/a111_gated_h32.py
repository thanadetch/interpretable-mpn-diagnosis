"""a111 — baseline gated-attention with a NARROWER bottleneck (hidden_dim=32).

The U-shape (underfit at low-DOF, val-overfit at the 128-wide bottleneck) was
never swept on the bottleneck WIDTH itself. A narrower bottleneck has fewer free
params in the 1280->H map (where the fold-specific val-overfit variance lives),
so it may sit lower on the U-shape and generalise better PAIRED. a111 (H=32) /
a112 (H=64) vs a104 (H=128 baseline) is a clean capacity-width sweep.
"""
from __future__ import annotations

from .a103_warmstart_gated import Model  # noqa: F401  (gated attention, configurable hidden_dim)

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=32, dropout=0.5, warm_start=False)
