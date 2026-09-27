"""a156 — Fully self-contained axis-attention (a155 with warm_start=False, learns v via backprop).

Strictest NO-NEW-DATA + ZERO-PRECOMPUTE version: no prototype file, no concept dir — the fibrosis
axis v, the attention sharpness λ, and the affine (w_s,c_s) are ALL learned by backprop from the
raw frozen features alone. Tests whether the per-fold prototype warm-start (the only remaining
"precompute", a gray area) is actually needed, or whether a self-contained aggregator on plain
`data` can clear GATE 1 by itself.

Expectation: learning a 1280-d axis from 857 batch-size-1 bags is hard (cf a129 underfit), so this
likely underfits — if so it JUSTIFIES the tiny prototype init (keeps a155 as the flagship); if it
clears GATE 1 it is the cleanest-possible novelty (nothing precomputed at all). Reads plain `data`.
"""
from __future__ import annotations
from .a155_axis_attention_density import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, warm_start=False)
