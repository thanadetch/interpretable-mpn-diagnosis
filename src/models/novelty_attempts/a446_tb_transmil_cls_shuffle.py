"""a446 — control for a445: the readout token is a real whole-ROI embedding from a DIFFERENT ROI.

Same architecture, same parameter count, same token count as a445. The only thing destroyed is the
correspondence between the readout token and the bag it reads. The substitute is drawn
deterministically from the bag's own fingerprint, so it is stable across epochs and at eval, and it
has the same marginal distribution and scale as the real field vector.

If a445's gain survives here, the gain does not come from the global view of THIS ROI — it comes
from the readout token merely varying across bags, and there is nothing about global context to
claim. This is the control that falsified the a340-a361 field-as-query family.
"""
from __future__ import annotations

from .two_branch_transmil import Model  # noqa: F401

KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="shuffle")
