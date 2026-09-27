"""a405 — Tri-alpha median ensemble: three branches at entmax alpha = 1.0 / 1.5 / 2.0, median combine.

The main configuration. See `tri_alpha_median.py` for the rationale, the check against the 424
modules already on disk, and why the median rather than the mean. Its control is a406, which is
identical in every way except that all three branches share alpha = 1.5, isolating whether the
sparsity DIVERSITY does the work or whether any three-branch ensemble would.
"""
from .tri_alpha_median import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, alphas=(1.0, 1.5, 2.0))
