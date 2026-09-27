"""cutmix_globalstd_noise - uniform CutMix + noise scaled by GLOBAL per-dim std (train-only).

cutmix_then_noise scales the noise by each BAG's own per-dim std (which varies bag to bag). This scales it
instead by the GLOBAL per-dim std computed once from the whole train pool, giving a stable, bag-independent
noise magnitude per channel. Tests whether a consistent noise scale beats a bag-adaptive one. Uniform
donors. `strength` = cutmix fraction. MPS-safe, deterministic given seed.
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
        self._gstd: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        allrows = []
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            f = item[0].float()
            chunks[g].append(f)
            allrows.append(f)
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        allc = torch.cat(allrows, dim=0)
        if allc.shape[0] > self.max_bank:
            allc = allc[torch.randperm(allc.shape[0])[: self.max_bank]]
        self._gstd = allc.std(dim=0, keepdim=True).clamp(min=1e-6)
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
        gstd = self._gstd.to(features.device, features.dtype)
        out = out + torch.randn_like(out) * (gstd * SIGMA)
        return out, float(label)
