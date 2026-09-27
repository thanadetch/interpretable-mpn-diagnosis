"""instance_dropout - randomly drop a fraction of patches per epoch (train-only).

Each patch is independently kept with probability (1 - strength). Forces the aggregator
to be robust to WHICH subset of tissue was sampled (a bag-level dropout). Target is
unchanged (the grade of a sub-sampled marrow region is the same diffuse density).

`strength` = drop probability in [0,1). 0 disables. Keep it gentle (<=0.3): the grade is
a holistic diffuse density, so dropping too many patches destroys the signal.

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
        self.p = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.p <= 0.0 or features.shape[0] < 2:
            return features, float(label)
        keep = torch.rand(features.shape[0], device=features.device) > self.p
        if int(keep.sum()) < 1:
            return features, float(label)  # never empty the bag
        return features[keep], float(label)
