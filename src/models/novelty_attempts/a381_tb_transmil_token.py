"""a381_tb_transmil_token — two-branch TransMIL variant.

TWO-BRANCH, FUSED IN THE SEQUENCE — the field view enters as an extra TOKEN alongside [CLS]
and the patches, with a learned modality embedding. Self-attention lets patches attend to the
field and the field attend to patches, instead of the two views meeting only at the head.
This is the transformer-native version of the same idea and the one most likely to beat a380.

    fuse='token'  field='roi'   base: TransMIL (a373)

Mechanism, controls and how to read the result: see ``two_branch_transmil.py``.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="token", field="roi")
