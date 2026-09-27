"""cutmix_then_dropout - compose within-grade CutMix with instance dropout (train-only).

Untested composition: cross-bag same-grade patch transplant (adds real inter-ROI diversity) followed
by instance-level dropout (within-bag regularisation, on-manifold). Both components are standard;
this pairs cutmix's diversity with dropout's regularisation instead of cutmix+noise.

    1. cutmix_within_grade @ ``strength``  (replace a fraction of patches with same-grade donors)
    2. instance_dropout @ 0.10             (drop 10% of the resulting patches)

`strength` = cutmix replacement fraction. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

DROP_RATE = 0.10


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
        bank: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        # step 1: within-grade cutmix
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            if self._bank is not None:
                g = int(round(float(label)))
                bank = self._bank.get(g)
                if bank is not None and bank.shape[0] >= 1:
                    n = features.shape[0]
                    k = min(max(1, int(round(self.s * n))), n)
                    dst = torch.randperm(n)[:k]
                    src = torch.randint(bank.shape[0], (k,))
                    out = features.clone()
                    out[dst] = bank[src].to(features.device, features.dtype)
        # step 2: instance dropout (keep at least a few patches)
        n = out.shape[0]
        if n >= 8:
            keep = max(4, int(round(n * (1.0 - DROP_RATE))))
            idx = torch.randperm(n)[:keep]
            out = out[idx]
        return out, float(label)
