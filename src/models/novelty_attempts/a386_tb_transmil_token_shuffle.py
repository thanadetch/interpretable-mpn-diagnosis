"""a386_tb_transmil_token_shuffle — DECISIVE CONTROL for a381.

Identical to a381 (fuse='token') except the extra token carries the whole-ROI field embedding of
a DIFFERENT, deterministically chosen ROI. Same architecture, same parameter count, same marginal
distribution and scale of the field vector — only the CORRESPONDENCE between the patch bag and
its own field view is destroyed.

    fuse='token'  field='shuffle'   base: TransMIL (a373)

KILL CONDITION, stated before running: if a386 lands within +/-0.005 test QWK of a381 on 2 of 3
backbones, then the field token's CONTENT does nothing and a381's gain is extra capacity, not
whole-field context. This is exactly how the a340-a361 FC-MIL family was falsified.

Mechanism and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="token", field="shuffle")
