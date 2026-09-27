"""bag_augment_noise - GROW the bag with extra uniform same-grade patches + noise (train-only).

A different axis from CutMix (which REPLACES patches, changing the mix but not the count): this APPENDS
extra real same-grade patches drawn uniformly from the grade bank, keeping every original patch. The bag
gets larger and its evidence pool more diverse, which — given the finding that the sparse-attention ASGAP
head thrives on uniform within-grade diversity — may give entmax more good options to attend to without
discarding any real evidence (CutMix's replacement is what collapses titan; pure addition keeps originals).
A small feature-noise step follows.

    add round(strength*N) uniform same-grade donor patches:  bag' = concat(features, donors)
    then add N(0, (per-dim std * SIGMA)^2) to every patch.                       label preserved.

`strength` = added patches as a fraction of bag size. Requires the train pool. MPS-safe, deterministic.
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
        k = max(1, int(round(self.s * n)))
        src = torch.randint(bank.shape[0], (k,))
        donors = bank[src].to(features.device, features.dtype)
        out = torch.cat([features, donors], dim=0)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
