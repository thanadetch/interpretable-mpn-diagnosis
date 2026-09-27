"""a463 - CONTROL for a462: the same operator with the similarity scale removed.

a462 has three scales (token / similar-set mean / bag mean). This keeps only token and bag mean,
so the medium scale collapses onto the coarse one. If a463 matches a462, the content-similarity
neighbourhood adds nothing and what PPEG contributes at ROI scale is plain bag-level smoothing --
which is the reading the a458/a459/a460 results already point to. If a462 wins, the similarity
neighbourhood is doing real work that grid adjacency was only approximating.

Gate: control, not a candidate. Report beside a462.
"""
from __future__ import annotations
from .two_branch_transmil import Model  # noqa: F401
KWARGS = dict(input_dim=1280, num_classes=1, fuse="cls", field="roi", pos_mode="roibag")
