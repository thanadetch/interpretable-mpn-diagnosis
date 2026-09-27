"""compose_drop_noise - stack instance-dropout THEN feature-noise (train-only). NEW: composition.

Prior augmentations were applied one at a time. This composes two: first drop a fraction of
patches (instance dropout at prob = strength), then add per-dim Gaussian noise (scale = 0.5*
strength * within-bag std) to the surviving patches. Tests whether stacking two mild regularisers
behaves differently from either alone.

`strength` drives both sub-augmentations. 0 disables. Target unchanged.
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
        self.s = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0:
            return features, float(label)
        # 1) instance dropout
        if features.shape[0] >= 2:
            keep = torch.rand(features.shape[0], device=features.device) > self.s
            if int(keep.sum()) >= 1:
                features = features[keep]
        # 2) feature noise (per-dim std scaled)
        std = features.std(dim=0, keepdim=True)
        features = features + torch.randn_like(features) * (0.5 * self.s * std)
        return features, float(label)
