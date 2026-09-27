"""patch_mixup_self - INTRA-bag manifold mixup of patch pairs (train-only). NEW: within-bag.

All prior mixup variants combine patches from DIFFERENT bags (cross-bag). This interpolates
pairs of patches WITHIN the same bag: each output patch is ``lam * h_i + (1-lam) * h_perm(i)``
with a single bag-level ``lam ~ Beta(strength, strength)`` and a random permutation of the
bag's own patches. It manufactures new on-manifold patches that lie between real patches of
the SAME ROI, so the grade is unchanged (no label noise) - a patch-level "manifold mixup"
that smooths the bag's empirical patch distribution without importing foreign-grade content.

`strength` = Beta concentration (->1 = lam near 0.5 = strong blend; small = near identity).
0 disables. Target unchanged. Permutation/size-invariant (output is a fixed-size convex
recombination of the bag), deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5):
        self.alpha = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.alpha <= 0.0 or features.shape[0] < 2:
            return features, float(label)
        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())
        perm = torch.randperm(features.shape[0], device=features.device)
        mixed = lam * features + (1.0 - lam) * features[perm]
        return mixed, float(label)
