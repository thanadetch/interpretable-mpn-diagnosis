# RETIRED — DO NOT RUN. The alpha lookup below (2.00 / 1.75 / 1.00 / 1.50) was chosen by
# sweeping alpha over the CV OUT-OF-FOLD TEST predictions of all 1,330 ROIs, so the table has
# seen every label it would later be scored on — under CV and under the single split alike,
# because the 259 test ROIs are a subset of those 1,330. All 36 runs were deleted 2026-08-19.
# The leak-free replacement is scripts/scale_alpha_clean.py, which refits the rule on each
# fold's VALIDATION split and applies it to that fold's test split.

"""a407 — entmax order looked up from the ROI's MEASURED um/px (no learning, zero extra params).

The main configuration. Rule, thresholds and the grade confound that it must survive are
documented in `scale_alpha.py`. Its control is a408, which applies the identical lookup to a
randomly permuted scale assignment; a407 is only interpretable next to it.
"""
from .scale_alpha import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
