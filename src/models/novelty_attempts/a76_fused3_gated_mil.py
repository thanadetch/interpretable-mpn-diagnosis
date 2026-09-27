"""a76 — 3-WAY fused gated-attention MIL (last fusion lever).

Same capacity-matched gated-attention aggregator as a68/a69, but on the 3-way
fused space Virchow2 (1280) + UNI2-h (1536) + TITAN (768) = 3584-d. Companion
single-backbone ablation = a69 (1280-d). Run on data_fused3 (built by
scripts/fuse_features.py --backbones virchow2 uni2 titan --out_root data_fused3).

Run:  --backbone virchow2 --data_root data_fused3 --novelty_id a76_fused3_gated_mil
Compare PAIRED vs a69 (single) across folds. Prior is low (2-way fusion tied,
TITAN weakest in C3) — this completes the fusion lever inventory.
"""
from __future__ import annotations

from .a68_fused_gated_mil import Model  # noqa: F401  (re-exported)

KWARGS = dict(
    input_dim=1280,            # ignored — trainer overwrites; see feature_dim_override
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    feature_dim_override=3584,  # Virchow2 1280 + UNI2-h 1536 + TITAN 768; run with --data_root data_fused3
)
