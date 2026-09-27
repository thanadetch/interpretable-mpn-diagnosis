"""a447 — second reference for a445: the readout token is the bag's own patch mean.

The patch mean is computable from the instances the model already sees, so it carries no
information the bag does not already contain — but it does vary across bags and it does correspond
to this bag. It therefore separates "the whole-ROI VIEW helps" from "a bag-consistent summary at
the readout position helps".
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="mean")
