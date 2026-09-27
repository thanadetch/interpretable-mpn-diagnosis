"""cutmix_proto_then_noise - FIXED-strength prototype CutMix THEN global feature noise (train-only).

The documented ASGAP-beating augmentation cutmix_then_noise uses UNIFORM same-grade donors. The study
champion cutmix_prototype_sharp showed prototype-weighted donors (sampled near the grade centroid, TAU=0.4)
beat uniform donors. This aug upgrades cutmix_then_noise's donor sampling to prototype-weighted while
keeping its two winning traits:
  - FIXED cutmix strength (the stochastic-strength variant proto_stoch_then_noise over-perturbed -> ep4
    under-training and titan collapse; fixed strength trains fully, cf. cutmix_then_noise ep15-40), and
  - a global feature-noise step (the element that let the sparse-attention ASGAP aggregator beat ABMIL).

    1. replace round(strength*N) patches with near-centroid (TAU=0.4) same-grade donors
    2. add N(0, (per-dim std * SIGMA)^2) to EVERY patch
    label preserved.

`strength` = cutmix fraction. SIGMA = noise level. Requires the train pool. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
SIGMA = 0.05


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
        bank, weights = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
        self._bank, self._w = bank, weights

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
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
