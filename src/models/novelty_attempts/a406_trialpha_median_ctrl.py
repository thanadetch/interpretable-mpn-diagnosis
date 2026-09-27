"""a406 — CONTROL for a405: three branches, median combine, but ALL at alpha = 1.5.

Same architecture, same parameter count, same combine rule, no sparsity diversity. If a405 beats
this, the alpha spread is responsible; if a405 only matches it, the gain is plain ensembling and
the tri-alpha story is dead. This row is what makes a405 interpretable, so the two must always be
run and reported together.
"""
from .tri_alpha_median import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, alphas=(1.5, 1.5, 1.5))
