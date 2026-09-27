"""a461 - a445 with BOTH mixing geometries: squarified 2-D PPEG and 1-D sequence convolution.

Measured separately, they win on different backbones:
    2-D (a445)  virchow2 .9675   uni2 .9371   titan .9616
    1-D (a458)  virchow2 .9577   uni2 .9511   titan .9581
so 2-D takes virchow2 and titan, 1-D takes uni2 by a clear margin (+.0140, 218 vs 206 correct,
G1 recall 40.8 -> 69.4). This module gives the model both paths and lets training weight them.

53,248 parameters in the position layer (44,032 + 9,216) against a445's 44,032.

PREDICTION, STATED BEFORE THE RUN: it loses. Nothing in this project has composed -- a448 and a449
(the global view in two roles at once) lost 2/3 and 3/3 this session, and every multi-mechanism
augmentation stack was at best equal to its parts. A loss here is the fourth independent
confirmation on 50 patients; a win would be the first exception and would need its own control.

Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", pos_mode="both")
