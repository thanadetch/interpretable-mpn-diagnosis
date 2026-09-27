"""cutmix_prototype_t06 - within-grade CutMix with prototype-weighted donor selection (train-only).

Standard cutmix_within_grade samples donor patches UNIFORMLY from the same-grade bank, which can
pull in noisy grade-boundary patches. Hypothesis: donors closer to the grade CENTROID (prototype)
are more representative of the grade, so weighting donor sampling toward the centroid should inject
cleaner grade signal. Everything else identical to cutmix_within_grade.

    weight(p) = softmax(-||p - centroid_g|| / tau)   # donors near the grade prototype favoured
    replace a `strength` fraction of the bag's patches with donors sampled by that weight.

`strength` = fraction replaced. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.6  # softmax temperature on distance-to-centroid


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None   # per-grade donor sampling weights
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
        bank: Dict[int, torch.Tensor] = {}
        weights: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            centroid = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - centroid, dim=1)            # [M]
            w = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            weights[g] = w
        self._bank = bank
        self._w = weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)          # prototype-weighted donors
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
