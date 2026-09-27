"""a451 — a445 with PPEG REMOVED.

Same question as a450, asked of the best-performing variant instead of the plain baseline: does the
whole-ROI-as-readout-token result depend on PPEG at all?
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", use_ppeg=False)
