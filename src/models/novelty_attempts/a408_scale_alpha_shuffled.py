# RETIRED — DO NOT RUN. The alpha lookup below (2.00 / 1.75 / 1.00 / 1.50) was chosen by
# sweeping alpha over the CV OUT-OF-FOLD TEST predictions of all 1,330 ROIs, so the table has
# seen every label it would later be scored on — under CV and under the single split alike,
# because the 259 test ROIs are a subset of those 1,330. All 36 runs were deleted 2026-08-19.
# The leak-free replacement is scripts/scale_alpha_clean.py, which refits the rule on each
# fold's VALIDATION split and applies it to that fold's test split.

"""a408 — CONTROL for a407: the same alpha lookup table applied to SHUFFLED scales.

Same rule, same marginal distribution of alphas over the cohort, correspondence between an ROI
and its own measured scale destroyed. If a408 matches a407 the measurement contributes nothing
and any gain belongs to the alpha marginal or to the scale-grade confound, not to the physical
scale. Always run and report it with a407.
"""
from .scale_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=True, seed=0)
