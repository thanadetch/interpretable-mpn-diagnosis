"""psemix_proto - PseMix with the paper's PHENOTYPE-STRATIFIED pseudo-bag division.

The registry's `psemix` module divides a bag into pseudo-bags by a plain random partition.
The original work instead divides by phenotype: instances are grouped by their cosine
distance to the bag prototype, and each pseudo-bag is then drawn *across* those groups, so
every pseudo-bag carries the whole bag's phenotype mix rather than an arbitrary subset. This
module pins that division ("ProtoDiv"), which is what the paper uses by default; everything
else - the pseudo-bag-level mixing and the area-aligned soft label - is inherited unchanged
from `psemix.py`.

    Liu et al., "Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole
    Slide Image Classification", IEEE TMI 2024 (arXiv:2306.16180).
"""
from __future__ import annotations

from .psemix import Augmentation as _PseMix

KWARGS = dict(strength=0.5)


class Augmentation(_PseMix):
    def __init__(self, strength: float = 0.5, k: int = 4, n_pheno: int = 4):
        super().__init__(strength=strength, k=k, division="proto", n_pheno=n_pheno)
