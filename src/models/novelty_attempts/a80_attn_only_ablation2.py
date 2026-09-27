"""a80 — ablation of a79: gated attention only (mode='attn_only').

= the locked baseline aggregation. a79 vs a80 isolates whether averaging the
three diverse grading-aligned mechanisms (attention + mean + coverage) reduces
variance / beats attention alone.
"""
from __future__ import annotations

from .a79_multimech_ensemble import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, mode="attn_only")
