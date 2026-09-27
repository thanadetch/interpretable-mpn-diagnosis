"""feature_dropout - drop random feature DIMENSIONS (train-only). NEW axis: dimension-level.

All prior augmentations act on PATCHES (drop/mix/resample rows). This acts on the feature
DIMENSIONS (columns): a random subset of the D feature dims is zeroed for the whole bag,
with inverted-dropout rescaling (1/(1-p)) so the expected magnitude is preserved. Standard
input-dropout regulariser, applied in the frozen feature space.

`strength` = dim drop probability in [0,1). 0 disables. Keep small (FM dims are entangled).
Permutation/size-invariant, deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.1)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.1):
        self.p = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.p <= 0.0 or features.shape[1] < 2:
            return features, float(label)
        keep = (torch.rand(features.shape[1], device=features.device) > self.p).float()
        if float(keep.sum()) < 1:
            return features, float(label)
        return features * keep / (1.0 - self.p), float(label)
