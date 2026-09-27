"""a383_tb_transmil_late_mean — two-branch TransMIL variant.

CONTROL for a380 — branch B is fed the MEAN PATCH embedding instead of the field view. Same
architecture, same parameters, NO information the patch bag did not already contain.

    fuse='late'  field='mean'   base: TransMIL (a373)

Mechanism, controls and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="late", field="mean")
