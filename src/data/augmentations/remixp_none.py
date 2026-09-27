"""remixp_none - ReMix `--mode None`: the REDUCE step with no bag mixing.

The original exposes None alongside the four mixing modes, and it is the ablation that
matters most here: it applies only the reduction of a bag to C prototypes, so any change
against the no-augmentation baseline is attributable to reduction alone rather than to the
mix-the-bag augmentation. On ROI bags of ~44 instances, reducing to C = 8 discards most of
the bag, which this variant measures directly.

    Yang et al., "ReMix: A General and Efficient Framework for Multiple Instance Learning
    based Whole Slide Image Classification", MICCAI 2022 (arXiv:2207.01805).
    https://github.com/TencentAILabHealthcare/ReMix
"""
from __future__ import annotations

from .remixp import Augmentation as _ReMixP

KWARGS = dict(strength=0.5)


class Augmentation(_ReMixP):
    def __init__(self, strength: float = 0.5, C: int = 8):
        super().__init__(strength=strength, C=C, rate=0.0, mode="none")
