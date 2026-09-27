"""a121 — ablation of a120: suppress=False = plain ABMIL (ignores bone/fib).

Isolates the bone-suppression / fibrosis-weighting ingredient: a120 vs a121 differ
ONLY by whether the two interpretable scalars bias the attention. a121 reads only
the first 1280 feature dims and reduces to the baseline gated attention.
"""
from __future__ import annotations

from .a120_bone_suppress_gated import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, suppress=False)
