"""a462 - PPEG replaced by a permutation-invariant multi-scale operator built for ROI bags.

PPEG defines coarse-to-fine by kernel width on a squarified grid. Every assumption behind that
grid was measured false on this cohort (see `_ROIScale` in two_branch_transmil). This defines the
same pyramid by bag fraction and feature similarity instead:

    fine    the token
    medium  mean of its N/4 most similar tokens
    coarse  mean of the bag

Permutation-invariant, no grid, no padding, identical behaviour at N=13 and N=112, and 3,072
parameters against PPEG's 44,032. Gates initialise at zero, so the module starts as the identity
and the run begins bit-identical to a445-without-PPEG.

Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", pos_mode="roiscale")
