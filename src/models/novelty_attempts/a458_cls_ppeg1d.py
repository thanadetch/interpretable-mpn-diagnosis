"""a458 - a445 with PPEG replaced by a 1-D depthwise convolution along the sequence.

The squarified grid invents vertical neighbours: measured on this cohort, a token's horizontal
neighbour in the ceil(sqrt(N)) grid is its true spatial neighbour 89.1% of the time, its vertical
neighbour 3.3% of the time. This keeps the real axis and drops the invented one. Everything else
-- including the squarify padding, so the sequence the attention layers see is byte-identical --
is unchanged, making kernel geometry the only variable.

Fewer parameters than PPEG (9,216 vs 44,032), so a win here is a win with less capacity; a loss is
confounded with capacity and must be read together with a460.

Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", pos_mode="ppeg1d")
