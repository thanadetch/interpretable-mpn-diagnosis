"""cutmix_2donor_noise - CutMix with 2-donor-averaged patches + feature noise (train-only).

Fills the last gap on the donor-diversity axis mapped in this study (NN ~ prototype < UNIFORM(peak) < FPS):
each replaced patch is the AVERAGE of TWO uniform same-grade donors. Averaging two donors mildly reduces
per-donor variance (a small step from single-uniform toward the centroid) without introducing a
centroid/prototype bias — a clean intermediate point between uniform single-donor CutMix (the sparse-attention
optimum) and prototype (concentrated) CutMix. Then the usual noise step. If it underperforms uniform, the
uniform optimum is confirmed from yet another direction.

    1. replace round(strength*N) patches, each with 0.5*(donor_a + donor_b), donors uniform same-grade
    2. add N(0, (per-dim std * SIGMA)^2) to every patch                       label preserved.

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
        a = torch.randint(bank.shape[0], (k,))
        b = torch.randint(bank.shape[0], (k,))
        donors = 0.5 * (bank[a] + bank[b])
        out = features.clone()
        out[dst] = donors.to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
