"""feature_noise - add seeded Gaussian noise to patch features (train-only).

For each patch: h' = h + strength * per_dim_std * N(0,1), where per_dim_std is the
within-bag standard deviation of each feature dimension (so the noise scale adapts to
each feature's natural spread). Smooths the input manifold and regularises the small head.

`strength` = relative noise scale. 0 disables. Keep small (<=0.2): frozen FM features
live on a learnt manifold, and isotropic noise pushes OFF that manifold (a known
weakness vs image-space augmentation) - so this is a heuristic regulariser.

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
        self.scale = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.scale <= 0.0 or features.shape[0] < 2:
            return features, float(label)
        std = features.std(dim=0, keepdim=True)  # [1, D] within-bag per-dim std
        noise = torch.randn_like(features) * (self.scale * std)
        return features + noise, float(label)
