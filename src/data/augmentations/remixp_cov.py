"""remixp_cov - ReMix `--mode cov` (reference implementation).

Pins one value of the original CLI's ``--mode`` flag for the whole run, as the paper's own
experiments do. See remixp.py for the mechanism and the honest deviation list.

    Yang et al., "ReMix: A General and Efficient Framework for Multiple Instance Learning
    based Whole Slide Image Classification", MICCAI 2022 (arXiv:2207.01805).
    https://github.com/TencentAILabHealthcare/ReMix
"""
from __future__ import annotations

from .remixp import Augmentation as _ReMixP

KWARGS = dict(strength=0.5)


class Augmentation(_ReMixP):
    def __init__(self, strength: float = 0.5, C: int = 8):
        super().__init__(strength=strength, C=C, rate=0.5, mode="cov")
