"""bag_union_subsample_noise - resample a bag from the union of two same-grade bags + noise (train-only).

Pools the current bag with one same-grade donor BAG, then subsamples back to the ORIGINAL size N so that
about strength*N patches come from the donor and (1-strength)*N from the original. Unlike bag_augment_noise
(which GREW the bag and catastrophically collapsed ASGAP), this keeps the bag size fixed = titan/ASGAP-safe.
Unlike flat-bank CutMix, donors come from a SINGLE real slide, preserving that slide's intra-ROI patch
correlations. A small all-patch noise step follows.

    donor_bag ~ same-grade bag ;  keep k=(1-strength)*N random originals + (strength*N) random donor patches
    concat -> bag of size N ;  add noise sigma=0.05                                    label preserved.

`strength` = donor fraction of the resampled bag. Requires the train pool (whole bags). MPS-safe.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bags: int = 400):
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
        kd = min(max(1, int(round(self.s * n))), n - 1)      # patches from donor
        ko = n - kd                                          # patches kept from original
        oi = torch.randperm(n)[:ko]
        di = torch.randint(donor.shape[0], (kd,))
        out = torch.cat([features[oi], donor[di]], dim=0)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
