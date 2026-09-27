"""a108 — ablation of a107: shrink_lambda=1.0 -> plain DIFFUSE mean-pool baseline.

Imports a107's Model with shrink_lambda=1.0. At lam=1 the dispersion is shrunk
fully to the bag mean dispersion vbar for every dimension, so the precision
weight w == 1 uniformly and the precision-weighted mean z_w == z EXACTLY — i.e.
the plain diffuse mean-pool with a byte-identical encoder, head and warm-start.

a107 (lam=0.5, precision reweight ON) vs a108 (lam=1.0, OFF) isolates EXACTLY the
active ingredient: "does inverse-within-bag-dispersion (precision) reweighting of
the diffuse pool beat the plain diffuse mean on PAIRED cross-fold Δ?".
forward uses ONLY 'features'. RAW logits.
"""
from __future__ import annotations

from .a107_precision_diffuse_pool import Model  # noqa: F401

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
