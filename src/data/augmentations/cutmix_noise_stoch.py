"""cutmix_noise_stoch - stochastic-policy version of cutmix_then_noise (train-only).

cutmix_then_noise uses a FIXED cutmix fraction and a FIXED noise level. This variant samples both
PER BAG from a range each call, exposing the model to a distribution of augmentation magnitudes
(the RandAugment / Beta-mixup idea) instead of a single operating point. If the fixed-strength champion
is near its ceiling, a richer stochastic policy may generalise slightly better.

  per bag:  s ~ U(s_lo, s_hi)   (cutmix fraction),   sigma ~ U(0, noise_hi)   (feature-noise level)
  step 1: within-grade CutMix @ s  (real same-grade donors)
  step 2: h += std * sigma * N(0,1)  (feature noise on the whole bag)

`strength` = the CENTRE of the cutmix-fraction range (range = strength +/- 0.1). Label preserved.
MPS-safe, deterministic given the global seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.75)

HALF_RANGE = 0.10   # cutmix fraction sampled in [strength-0.1, strength+0.1]
NOISE_HI = 0.07     # per-bag noise level sampled in [0, NOISE_HI]


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.75, max_bank: int = 20000):
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
        # per-bag stochastic cutmix fraction
        s = float(torch.empty(1).uniform_(self.s - HALF_RANGE, self.s + HALF_RANGE).clamp_(0.05, 0.95).item())
        k = min(max(1, int(round(s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        # per-bag stochastic noise level
        sigma = float(torch.empty(1).uniform_(0.0, NOISE_HI).item())
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * sigma)
        return out, float(label)
