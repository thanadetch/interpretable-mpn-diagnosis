"""bag_mixup_noise - per-patch MixUp toward a whole same-grade donor BAG + noise (train-only).

Different donor SOURCE than the flat patch-bank CutMix: here a single same-grade donor BAG is sampled from
the pool and each patch of the current bag is blended toward a random patch of that donor bag by a per-patch
weight lam ~ U(0, strength). Drawing the donor from ONE bag preserves that bag's intra-ROI patch correlations
(vs a flat bank that mixes patches from many slides), testing whether coherent per-bag donors help. A small
all-patch noise step follows (the ASGAP-helping regulariser).

    donor_bag ~ same-grade bag from pool ;  for each patch i: j = random patch of donor_bag
    h_i' = (1-lam_i)*h_i + lam_i*donor_bag[j] ,  lam_i ~ U(0, strength) ;  then noise sigma=0.05
    label preserved.

`strength` = max mixup weight. Requires the train pool (whole bags). MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bags: int = 400):
        self.s = float(strength)
        self.max_bags = int(max_bags)
        self._bags: Optional[Dict[int, List[torch.Tensor]]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        bags: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            bags[g].append(item[0].float())
        for g in bags:
            if len(bags[g]) > self.max_bags:
                idx = torch.randperm(len(bags[g]))[: self.max_bags].tolist()
                bags[g] = [bags[g][j] for j in idx]
        self._bags = bags

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bags is None:
            return features, float(label)
        g = int(round(float(label)))
        lst = self._bags.get(g)
        if not lst:
            return features, float(label)
        donor = lst[int(torch.randint(len(lst), (1,)).item())].to(features.device, features.dtype)
        n = features.shape[0]
        j = torch.randint(donor.shape[0], (n,))
        d = donor[j]
        lam = torch.empty(n, 1, device=features.device, dtype=features.dtype).uniform_(0.0, self.s)
        out = (1.0 - lam) * features + lam * d
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
