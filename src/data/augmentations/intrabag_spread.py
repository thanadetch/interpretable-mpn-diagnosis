"""intrabag_spread - jitter the WITHIN-BAG patch spread around the centroid (train-only).
NEW: variance-axis (mean preserved).

Every patch is re-expressed as ``mean + u * (patch - mean)`` with a single bag-level factor
``u = 1 + strength * N(0,1)`` (clamped > 0.1). This expands (u>1) or contracts (u<1) how
spread-out the patches are around the bag centroid while leaving the centroid - and hence the
attention-weighted mean the head mostly reads - unchanged. Tests robustness to density-spread
heterogeneity (some ROIs have more within-bag texture variation than others) without touching
the bag's mean representation or its label.

`strength` = std of the spread factor u. 0 disables. Target unchanged.
Permutation/size-invariant, deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2):
        self.s = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or features.shape[0] < 2:
            return features, float(label)
        mean = features.mean(dim=0, keepdim=True)
        u = 1.0 + self.s * float(torch.randn(1).item())
        u = max(0.1, u)
        return mean + u * (features - mean), float(label)
