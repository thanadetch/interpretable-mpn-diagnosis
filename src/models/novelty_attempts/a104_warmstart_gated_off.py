"""a104 — ablation of a103: warm_start=False (= plain baseline gated attention).

a103 (warm head) vs a104 (default head) isolates whether warm-starting the
frontier model's head along the fibrosis axis helps paired generalisation.
"""
from __future__ import annotations

from .a103_warmstart_gated import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=False)
