"""remix_covary - ReMix with the 'covary' mix-the-bag op PINNED (train-only, feature-space).

The registry's `remix` module draws one of ReMix's four latent augmentations per bag. The
original work instead selects a SINGLE op for a whole training run (its CLI exposes
`--mode {append,replace,interpolate,cov}`), so a like-for-like comparison against the paper
needs one run per op. This module pins op=3 (covary, the paper's `--mode cov`): zero-mean
Gaussian noise scaled by the PER-GRADE prototype standard deviation is added to the reduced
bag's prototypes, i.e. the bag is perturbed along the directions in which that grade actually
varies (the paper's "semantic variation" direction).

Everything else - the k-means "reduce" step, the per-grade prototype bank, and the meaning of
`strength` - is inherited unchanged from `remix.py`.

    Yang et al., "ReMix: A General and Efficient Framework for Multiple Instance Learning based
    Whole Slide Image Classification", MICCAI 2022 (arXiv:2207.01805).
"""
from __future__ import annotations

from .remix import Augmentation as _ReMix

KWARGS = dict(strength=0.5)


class Augmentation(_ReMix):
    def __init__(self, strength: float = 0.5, C: int = 8, max_bank: int = 20000):
        super().__init__(strength=strength, C=C, max_bank=max_bank, op=3)
