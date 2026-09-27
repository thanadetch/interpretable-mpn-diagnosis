"""bag_bootstrap - resample the bag's patches WITH replacement (train-only).

Draws M = round(strength * N) patches uniformly WITH replacement from the bag's own N
patches, forming a bootstrap resample of the same tissue. Each epoch the aggregator sees
a slightly different empirical patch distribution (some patches duplicated, some absent),
a bagging-style regulariser. Target unchanged (same grade, same tissue).

`strength` = resample ratio M/N (default 1.0 = same size). 0 disables.

Permutation/size-invariant, deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=1.0)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 1.0):
        self.ratio = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.ratio <= 0.0 or features.shape[0] < 2:
            return features, float(label)
        n = features.shape[0]
        m = max(1, int(round(self.ratio * n)))
        idx = torch.randint(n, (m,), device=features.device)  # with replacement
        return features[idx], float(label)
