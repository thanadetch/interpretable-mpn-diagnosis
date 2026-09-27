"""cutmix_noise_donors - uniform CutMix with noise on the DONOR patches only (train-only).

cutmix_then_noise adds noise to ALL patches; proto_stoch_noise added it to the KEPT patches (worse). This
tests the third placement: noise ONLY the transplanted DONOR patches, leaving the bag's real original
patches pristine. Rationale: donors are exact copies from the bank (identical to real patches elsewhere),
so jittering them adds donor diversity / breaks exact duplication, while the original evidence stays clean.
Uniform donors (ASGAP-preferred). `strength` = cutmix fraction. MPS-safe, deterministic.
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
        donors = bank[src].to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        donors = donors + torch.randn_like(donors) * (std * SIGMA)   # noise donors only
        out = features.clone()
        out[dst] = donors
        return out, float(label)
