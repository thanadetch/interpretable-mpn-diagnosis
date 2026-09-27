"""a110 — ABLATION of a109 (a107_precision_diffuse_pool): shrink_lambda=1.0
-> plain DIFFUSE mean-pool baseline.

Imports a109's Model with shrink_lambda=1.0. At lam=1 the dispersion is shrunk
fully to the bag mean dispersion vbar for every dimension, so the precision
weight w == 1 uniformly and the precision-weighted mean z_w == z EXACTLY — i.e.
the plain diffuse mean-pool with a BYTE-IDENTICAL encoder, head and warm-start.

This flips EXACTLY one thing — the inverse-within-bag-dispersion precision
reweight (shrink_lambda 0.5 -> 1.0) — and nothing else. Everything else (the
1280->128 encoder, the Linear(128,1) head, the seed=2 warm-start, dropout, eps)
is identical to a109.

a109 (lam=0.5, precision reweight ON) vs a110 (lam=1.0, OFF) isolates EXACTLY the
active ingredient: "does inverse-within-bag-dispersion (precision) reweighting of
the diffuse pool beat the plain diffuse mean on PAIRED cross-fold Δ?".
forward uses ONLY 'features'. RAW logits.
"""
from __future__ import annotations

from .a109_a107_precision_diffuse_pool import Model  # noqa: F401

KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    shrink_lambda=1.0,     # OFF: w == 1 -> plain diffuse mean-pool baseline
    warm_start=True,
    prototype_path=None,
    eps=1e-6,
    clamp_output=False,
)
