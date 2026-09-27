"""a448 — the MIXED variant of a445: the learned CLS is kept and biased by the global view.

    a381    seq = [CLS] + [field] + patches      field is an extra token, competes for attention
    a445    seq = [field] + patches              field REPLACES the readout token
    a448    seq = [CLS + field] + patches        field BIASES the readout token   <- this one

a445's readout token is recomputed from the ROI on every forward pass and carries nothing learned
of its own; a448 keeps a component that is constant across bags (the learned CLS) alongside the
per-bag component. If a448 lands between plain TransMIL and a445, the effect scales with how much
of the readout token is data-dependent.

Parameter count is unchanged (2,805,249); unlike a445 the ``cls_token`` is live here.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="clsadd", field="roi")
