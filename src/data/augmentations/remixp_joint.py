"""remixp_joint - ReMix `--mode joint` (reference implementation).

Pins one value of the original CLI's ``--mode`` flag for the whole run, as the paper's own
experiments do. See remixp.py for the mechanism and the honest deviation list.
  The README recommends a lower rate for joint (0.1-0.2) than the 0.5 default;
  this variant uses 0.2.

    Yang et al., "ReMix: A General and Efficient Framework for Multiple Instance Learning
    based Whole Slide Image Classification", MICCAI 2022 (arXiv:2207.01805).
    https://github.com/TencentAILabHealthcare/ReMix
"""
from __future__ import annotations

from .remixp import Augmentation as _ReMixP

KWARGS = dict(strength=0.5)


class Augmentation(_ReMixP):
    def __init__(self, strength: float = 0.5, C: int = 8):
        super().__init__(strength=strength, C=C, rate=0.2, mode="joint")
