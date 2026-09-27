"""feature_scale - per-dimension MULTIPLICATIVE jitter (train-only). NEW: multiplicative.

feature_noise adds Gaussian noise (location jitter); this MULTIPLIES each feature dimension
by a shared random gain ``1 + strength * N(0,1)`` (scale jitter), applied identically across
all patches in the bag (a consistent re-gaining of the frozen feature space, not per-patch
corruption). Tests whether the head is robust to small affine re-scalings of the embedding
axes. Distinct mechanism from additive noise and from dim-dropout (which hard-zeros dims).

`strength` = std of the per-dim multiplicative gain. 0 disables. Target unchanged.
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
        gain = 1.0 + self.s * torch.randn(1, features.shape[1], device=features.device, dtype=features.dtype)
        return features * gain, float(label)
