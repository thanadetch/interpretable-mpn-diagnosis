"""cutmix_stochnoise - uniform CutMix + per-bag STOCHASTIC noise level (train-only).

The sigma sweep showed the ASGAP win is a plateau over noise level (test 0.9684/0.9695/0.9673 at
sigma 0.03/0.05/0.08). This variant exploits that: instead of a single fixed sigma it draws the noise
level PER BAG from U(LO, HI) spanning the whole good plateau, exposing the model to every good noise level
each epoch. Uniform donors (the ASGAP-preferred diversity). The aim is to make the ASGAP-beats-ABMIL win
even more robust (never depending on one lucky sigma).

    per bag:  replace round(strength*N) patches w/ uniform same-grade donors;
              sigma ~ U(LO, HI);  add N(0,(per-dim std * sigma)^2) to every patch.   label preserved.

`strength` = cutmix fraction. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
LO = 0.02
HI = 0.08


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
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        sigma = float(torch.empty(1).uniform_(LO, HI).item())
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * sigma)
        return out, float(label)
