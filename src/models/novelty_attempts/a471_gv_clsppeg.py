"""a471 — TransMIL + global view, placement variant: a445 แต่ให้ global เข้า PPEG

Part of a PRE-REGISTERED batch of four placements (a468–a471) announced together and reported
together regardless of outcome. Fifteen earlier variants around a445 all failed to beat it
consistently, so with a locked 259-ROI test set any single winner found by scanning must be
treated as a hypothesis, not a result.

fuse='clsppeg'. Mechanism and controls: see ``two_branch_transmil.py``.
Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="clsppeg", field="roi")
