"""patch_dropout - random instance (patch) dropout at the bag level (train-only).

The canonical bag-level MIL augmentation: each epoch, drop a random fraction of the bag's patches
before aggregation, forcing the permutation-invariant head to be robust to missing evidence and to
not over-rely on any single high-attention patch. Unlike CutMix (replaces patches with donors) or
noise (perturbs features), this changes only the multiset SIZE/composition — no synthetic features,
no cross-bag label risk, so the grade label is trivially preserved.

    per bag:  p ~ U(strength-0.1, strength+0.1);  keep round((1-p)*N) patches (>= MIN_KEEP), drop the rest.

`strength` = centre of the drop-fraction range. No train pool needed. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)

HALF_RANGE = 0.10
MIN_KEEP = 4


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3, max_bank: int = 20000):
        self.s = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        n = features.shape[0]
        if self.s <= 0.0 or n <= MIN_KEEP:
            return features, float(label)
        p = float(torch.empty(1).uniform_(self.s - HALF_RANGE, self.s + HALF_RANGE).clamp_(0.0, 0.9).item())
        keep = max(MIN_KEEP, int(round((1.0 - p) * n)))
        if keep >= n:
            return features, float(label)
        idx = torch.randperm(n)[:keep]
        return features[idx], float(label)
