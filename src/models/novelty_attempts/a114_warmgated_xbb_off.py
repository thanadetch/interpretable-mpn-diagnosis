"""a114 — ablation of a113: warm_start=False = plain ABMIL (any backbone).

The per-backbone baseline for the cross-backbone breadth test. Identical code to
a113 with the head warm-start turned OFF, so a113 vs a114 differ by EXACTLY the
classifier-head init along the seed=2 train fibrosis axis. Run both at seed=2 on
{virchow2, uni2, titan} and read Δ = warm - plain per backbone.
"""
from __future__ import annotations

from .a113_warmgated_xbb import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=False)
