"""beta_mixup - per-patch Beta-weighted MixUp toward a same-grade donor (train-only).

Classic MixUp (Zhang et al. 2018) adapted to MIL features: blend each patch with a random same-grade
donor using a Beta(alpha, alpha) weight. With small alpha the Beta mass concentrates near 0 and 1, so
most patches stay close to an endpoint (mild, mostly-identity or mostly-donor) while a few interpolate —
a richer, softer distribution than a fixed 0.5 blend. Same-grade donors keep the label valid.

    lambda ~ Beta(alpha, alpha)   per patch
    h' = lambda * h + (1 - lambda) * donor_same_grade      label preserved.

`strength` = alpha (Beta concentration; smaller = milder, mass at the endpoints). Requires the train pool.
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2, max_bank: int = 20000):
        self.a = float(strength)
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
        if self.a <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        donor = bank[torch.randint(bank.shape[0], (n,))].to(features.device, features.dtype)
        beta = torch.distributions.Beta(self.a, self.a)
        lam = beta.sample((n, 1)).to(features.device, features.dtype)
        return lam * features + (1.0 - lam) * donor, float(label)
