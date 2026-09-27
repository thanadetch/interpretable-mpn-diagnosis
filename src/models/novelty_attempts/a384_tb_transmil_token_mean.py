"""a384_tb_transmil_token_mean — two-branch TransMIL variant.

CONTROL for a381 — the extra token carries the mean patch embedding, not the field view.

    fuse='token'  field='mean'   base: TransMIL (a373)

Mechanism, controls and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="token", field="mean")
