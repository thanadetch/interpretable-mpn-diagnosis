"""a450 — plain TransMIL with PPEG REMOVED.

PPEG squarifies the instance sequence into a ceil(sqrt(N))^2 grid and mixes neighbours with three
depthwise convolutions. On this cohort that grid is not the ROI's real layout: a 5x8 patch grid is
reshaped to 7x7, and OD filtering has already punched holes in the sequence, so PPEG's "neighbours"
are largely not neighbours in the tissue. This module removes it to measure what it contributes.

Reading the result: a tie means the positional component of TransMIL does nothing here; a loss
means PPEG helps for reasons OTHER than position (it is also a local-mixing operator with its own
parameters); a win means the fake grid was actively harmful.
"""
from __future__ import annotations

from .a373_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, use_ppeg=False)
