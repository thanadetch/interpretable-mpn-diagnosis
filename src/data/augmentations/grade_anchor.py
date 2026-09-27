"""grade_anchor - inject jittered grade-prototype anchor patches into the bag (train-only).

A different lever from CutMix (which REPLACES patches) and noise (which perturbs them): here we ADD a few
synthetic "anchor" patches equal to the grade prototype (centroid) plus small per-dim jitter, without
removing any real patch. The anchors reinforce the grade signal in the bag while preserving all original
evidence, and the jitter keeps them from being identical duplicates.

    add k = round(strength * N) anchors:  a = centroid_g + eps * per_dim_std_g * N(0,1)
    bag' = concat(original patches, anchors)      label preserved.

`strength` = number of anchors as a fraction of bag size. Requires the train pool. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)

JITTER = 0.5   # anchor jitter as a fraction of the grade's per-dim std


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._cent: Optional[Dict[int, torch.Tensor]] = None
        self._std: Optional[Dict[int, torch.Tensor]] = None
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
        cent, std = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cent[g] = allp.mean(dim=0, keepdim=True)               # [1, D]
            std[g] = allp.std(dim=0, keepdim=True).clamp(min=1e-6)  # [1, D]
        self._cent, self._std = cent, std

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._cent is None:
            return features, float(label)
        g = int(round(float(label)))
        c = self._cent.get(g); sd = self._std.get(g)
        if c is None:
            return features, float(label)
        c = c.to(features.device, features.dtype); sd = sd.to(features.device, features.dtype)
        n = features.shape[0]
        k = max(1, int(round(self.s * n)))
        anchors = c + JITTER * sd * torch.randn(k, features.shape[1], device=features.device, dtype=features.dtype)
        return torch.cat([features, anchors], dim=0), float(label)
