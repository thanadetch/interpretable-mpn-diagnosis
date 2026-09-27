"""a459 - a445 with PPEG's kernel pyramid rescaled from gigapixel bags to ROI bags.

TransMIL's 7/5/3 kernels assume N ~ 10^4 patches (grid ~100 wide), where a 7-kernel covers 7% of
the field. Measured here: median N=40 gives S=7, and 91.0% of the 1330 ROIs have S <= 7 -- so the
7-kernel spans the WHOLE grid and the coarse/medium/fine pyramid collapses into three bag-level
averages. 5/3/1 restores a real ordering at S=7 (71% / 43% / 14% of the grid).

This is the direct test of "PPEG is shaped for WSI, not for ROI". 19,456 parameters vs 44,032.

Gate: val QWK AND test QWK above a445 on the same backbone.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", pos_mode="ppeg_small")
