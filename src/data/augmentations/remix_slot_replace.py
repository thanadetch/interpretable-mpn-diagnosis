"""remix_slot_replace - ReMix 'replace' with SLOT-ALIGNED prototypes (the paper's setting).

`remix_replace` pins the op but still draws donor prototypes from a flat same-grade pool. The
original work matches prototypes across bags, so op 'replace' pairs corresponding prototype
slots rather than arbitrary ones. This module adds that alignment: all training patches are
clustered once, and prototype k of every bag is that bag's mean over global cluster k.

    Yang et al., "ReMix: A General and Efficient Framework for Multiple Instance Learning based
    Whole Slide Image Classification", MICCAI 2022 (arXiv:2207.01805).
"""
from __future__ import annotations

from .remix import Augmentation as _ReMix

KWARGS = dict(strength=0.5)


class Augmentation(_ReMix):
    def __init__(self, strength: float = 0.5, C: int = 8, max_bank: int = 20000):
        super().__init__(strength=strength, C=C, max_bank=max_bank, op=1, align="global")
