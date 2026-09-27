"""a69 — SINGLE-backbone ablation of a68 (isolates the fusion contribution).

Identical architecture to a68 (the baseline gated-attention aggregator), but
``feature_dim_override=1280`` so it runs on single-backbone Virchow2 features.
a68 vs a69, evaluated PAIRED across the same patient folds, isolates EXACTLY
the fusion of complementary backbones (Virchow2 1280 + UNI2-h 1536 = 2816) —
the only difference between the two is the input feature space.

Run:  --backbone virchow2 --data_root data --novelty_id a69_single_gated_mil
This should track the locked `simple` baseline (same aggregator); a68 (fused)
beating a69 (single) on most folds = fusion adds grade signal.
"""
from __future__ import annotations

from .a68_fused_gated_mil import Model  # noqa: F401  (re-exported)

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    feature_dim_override=1280,  # single-backbone Virchow2; run with --data_root data
)
