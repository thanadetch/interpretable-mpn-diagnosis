"""a380_tb_transmil_late — two-branch TransMIL variant.

TWO-BRANCH TransMIL, LATE FUSION — the literal two-branch design. Branch A = TransMIL over
the ~44 non-resized patch tokens; branch B = an MLP over the whole-ROI (resized) embedding.
Branch B's head is zero-initialised, so training starts bit-identical to single-branch
TransMIL (a373) and any gain has to be learned.

    fuse='late'  field='roi'   base: TransMIL (a373)

Mechanism, controls and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="late", field="roi")
