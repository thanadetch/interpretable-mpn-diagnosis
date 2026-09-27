"""a452 — a381 with PPEG REMOVED.

Completes the PPEG on/off sweep across the three ways the field view can enter TransMIL:

    a373 / a450   no field         PPEG on / off
    a381 / a452   field as an extra token in the sequence      PPEG on / off   <- this one
    a445 / a451   field as the readout token                   PPEG on / off

Measured so far: removing PPEG costs plain TransMIL .011-.031 and costs a445 up to .074, so the
field-as-readout mechanism depends on the local mixing much more than the baseline does. This run
asks whether the field-as-extra-token mechanism depends on it too.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="token", field="roi", use_ppeg=False)
