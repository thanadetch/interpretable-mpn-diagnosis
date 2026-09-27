"""cutmix_then_tnoise - uniform CutMix + Student-t (df=3, heaviest tail) feature noise (train-only).

Noise-DISTRIBUTION test of the ASGAP-winning recipe cutmix_then_noise (uniform CutMix + GAUSSIAN noise).
This swaps the Gaussian for a Student-t (df=3) of the same per-dim scale: heavier tails mean most jitters are small
but occasional ones are large, a different regularisation profile. Uniform donors (ASGAP-preferred).

    1. replace round(strength*N) patches with uniform same-grade donors
    2. add Laplace(0, b) noise per dim with b = per-dim std * SIGMA / sqrt(2)  (so variance matches Gaussian sigma)
    label preserved.

`strength` = cutmix fraction. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
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
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        df = 3.0
        t = torch.distributions.StudentT(df).sample(out.shape).to(out.device, out.dtype)
        t = t * ((df - 2.0) / df) ** 0.5   # unit variance
        return out + t * (std * SIGMA), float(label)
