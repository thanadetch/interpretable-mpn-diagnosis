"""prototype_shrink - per-patch shrinkage toward the grade prototype (train-only, per-bag).

A denoising-style augmentation, orthogonal to the cutmix family: no patch swapping and no cross-bag
sampling. Each patch is pulled a random fraction of the way toward its grade's centroid (prototype),

    h'_i = h_i + a_i * (centroid_g - h_i),   a_i ~ Uniform(0, strength)   (per patch)

which shrinks within-grade variance toward the class prototype (a feature-space analogue of shrinkage
/ label-consistent denoising). Because it only uses the precomputed grade centroid and the bag's own
patches, it is PER-BAG (no cross-bag mixing) and therefore safe on slide-level TITAN, where cross-bag
cutmix collapses. Label is preserved.

`strength` = max shrink fraction toward the centroid. Requires the train pool (for centroids).
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._centroid: Optional[Dict[int, torch.Tensor]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            chunks[g].append(item[0].float())
        centroid: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            centroid[g] = allp.mean(dim=0, keepdim=True)   # [1, D]
        self._centroid = centroid

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._centroid is None:
            return features, float(label)
        g = int(round(float(label)))
        c = self._centroid.get(g)
        if c is None:
            return features, float(label)
        c = c.to(features.device, features.dtype)
        n = features.shape[0]
        a = torch.rand(n, 1, device=features.device, dtype=features.dtype) * self.s  # per-patch shrink
        out = features + a * (c - features)
        return out, float(label)
