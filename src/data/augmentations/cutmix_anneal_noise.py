"""cutmix_anneal_noise - champion CutMix ANNEALED OFF over training + noise (train-only).

The registry's untouched dimension is TIME: every one of the ~147 modules applies a fixed, i.i.d.-per-bag
transformation for all 50 epochs. That means the champion optimises the weights for a distribution of
80%-synthetic bags from the first step to the last - a distribution that never occurs at evaluation, where
bags are real ROIs. This module removes exactly that mismatch at the end of training:

    strength_t = ``strength`` * max(0, 1 - epoch / ANNEAL_EPOCHS)     (then feature_noise @ 0.05)

so training starts as the champion and finishes on (nearly) untouched real bags, letting the last epochs
fine-tune on the evaluation distribution while keeping the regularisation that the early epochs need.
This is the standard "turn the augmentation off at the end" recipe and it has never been tested here.

Epochs are counted from the number of calls (one call per training bag), so no trainer signal is needed:
epoch = calls // len(train_pool). Note best-epoch selection may land mid-schedule - that is part of the
test, not a flaw. `strength` = the initial cutmix fraction. MPS-safe, deterministic given the global seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

ANNEAL_EPOCHS = 30
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._n = 1
        self._calls = 0
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        self._n = max(1, len(pool))
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            chunks[int(round(float(item[1])))].append(item[0].float())
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        epoch = self._calls // self._n
        self._calls += 1
        s_t = self.s * max(0.0, 1.0 - epoch / float(ANNEAL_EPOCHS))
        out = features
        if s_t > 0.0 and self._bank is not None:
            bank = self._bank.get(int(round(float(label))))
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(s_t * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
