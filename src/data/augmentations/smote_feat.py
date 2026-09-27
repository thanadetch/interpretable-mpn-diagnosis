"""smote_feat - SMOTE-style feature interpolation toward a same-grade nearest neighbour (train-only).

A genuinely different family from CutMix (discrete replacement) and noise (random jitter): SMOTE
(Chawla et al. 2002) synthesises on-manifold samples by interpolating between a point and one of its
same-class nearest neighbours. Here, for every patch we sample a handful of same-grade candidates from
the train pool, pick the NEAREST, and move the patch a random fraction toward it:

    nn = argmin_{c in candidates(grade)} ||h - c||
    h' = h + lambda * (nn - h),   lambda ~ U(0, strength)   (per patch)

Nearest-neighbour interpolation stays on the local same-grade manifold (unlike blending toward a random
distant donor), producing realistic synthetic grade-consistent patches. `strength` = max interpolation
fraction. Label preserved. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)

N_CAND = 20   # same-grade candidates sampled per patch; nearest is chosen


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
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
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    @torch.no_grad()
    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 2:
            return features, float(label)
        bank = bank.to(features.device, features.dtype)
        n = features.shape[0]
        m = min(N_CAND, bank.shape[0])
        # sample m same-grade candidates per patch -> [n, m, D]
        idx = torch.randint(bank.shape[0], (n, m), device=features.device)
        cand = bank[idx]                                          # [n, m, D]
        d = torch.norm(cand - features.unsqueeze(1), dim=2)       # [n, m]
        nn = cand[torch.arange(n, device=features.device), d.argmin(dim=1)]  # [n, D] nearest
        lam = torch.rand(n, 1, device=features.device, dtype=features.dtype) * self.s
        return features + lam * (nn - features), float(label)
