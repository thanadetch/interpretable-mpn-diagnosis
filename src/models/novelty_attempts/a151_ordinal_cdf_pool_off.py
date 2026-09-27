"""a151 — ablation of a150 (OPAL): per_patch_cdf=False (pool THEN CDF = mean density → ordinal CDF).
a150 (CDF-then-pool, focal/diffuse-sensitive) vs a151 (pool-then-CDF, mean-only) isolates whether
the per-patch ordinal nonlinearity adds grade signal over the mean — i.e. another nonlinear
mean-sufficiency test. Reads plain `data`.
"""
from __future__ import annotations
from .a150_ordinal_cdf_pool import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, per_patch_cdf=False, warm_start=True)
