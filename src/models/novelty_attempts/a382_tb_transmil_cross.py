"""a382_tb_transmil_cross — two-branch TransMIL variant.

TWO-BRANCH, CROSS-ATTENTION — the field vector queries the contextualised patch tokens and
the result is fused with the [CLS] descriptor. Zero-initialised, starts as a373.

    fuse='cross'  field='roi'   base: TransMIL (a373)

Mechanism, controls and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="cross", field="roi")
