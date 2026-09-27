"""a385_tb_transmil_token_patient — two-branch TransMIL variant.

PATIENT-CONTEXT variant of a381 — the extra token is the leave-one-out mean field view of the
SAME patient's other ROIs. Transductive at the patient level (images only, never labels).

    fuse='token'  field='patient'   base: TransMIL (a373)

Mechanism, controls and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="token", field="patient")
