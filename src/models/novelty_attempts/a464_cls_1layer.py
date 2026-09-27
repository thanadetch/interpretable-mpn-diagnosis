"""a464 - a445 with ONE round of attention instead of two.

Measured on the a445 checkpoints: the readout token's attention over the patches is MORE
concentrated after layer 1 than after layer 2 on every backbone (effective-N / N = .515 / .630 /
.261 vs .828 / .896 / .345). The second round is therefore spreading a selection the first round
already made. This reads out directly after layer 1.

PPEG sits between the two rounds and edits patch tokens only; with a single round it could never
reach the readout, so it is dropped rather than kept inert. Parameter count falls by the whole
second block (~1.05M) plus PPEG (44K): 1,710,721 against a445's 2,805,249.

PREDICTION, STATED FIRST: ties or loses slightly. A tie is the interesting outcome -- it would mean
a field-initialised readout needs one round of attention, not two, and halves the model.

Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", n_layers=1)
