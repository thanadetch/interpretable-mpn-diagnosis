"""proto_stoch_noise - stochastic prototype CutMix + stochastic noise on kept patches (train-only).

Combines the three elements that individually helped, all stochastic per bag:
  - prototype-weighted donors (clean grade signal, TAU=0.4),
  - per-bag stochastic cutmix fraction (the proto_stoch plateau, not a single point),
  - per-bag stochastic feature noise added ONLY to the kept (non-donor) patches (keeps transplanted
    donors pristine while regularising the rest, per cutmix_proto_noise_orig).

    per bag:  s ~ U(strength-0.1, strength+0.1),   sigma ~ U(0, NOISE_HI)
    replace round(s*N) patches with prototype donors; add std*sigma noise to the kept patches only.

`strength` = centre of the cutmix-fraction range. Label preserved. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
HALF_RANGE = 0.10
NOISE_HI = 0.06


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
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
        weights: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
        self._bank = bank
        self._w = weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        s = float(torch.empty(1).uniform_(self.s - HALF_RANGE, self.s + HALF_RANGE).clamp_(0.05, 0.95).item())
        k = min(max(1, int(round(s * n))), n)
        perm = torch.randperm(n)
        dst, keep = perm[:k], perm[k:]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        if keep.numel() > 0:
            sigma = float(torch.empty(1).uniform_(0.0, NOISE_HI).item())
            std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
            out[keep] = out[keep] + torch.randn_like(out[keep]) * (std * sigma)
        return out, float(label)
