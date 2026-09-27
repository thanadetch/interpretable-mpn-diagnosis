"""rand_aug_policy - RandAugment-style stochastic policy over the two winning augmentations (train-only).

The search found two complementary winners: uniform-CutMix+noise (helps the patch-level backbones
virchow2/uni2, esp. ASGAP) and subspace_noise (the only thing that helps slide-level TITAN). No single
FIXED augmentation helps all three. This applies a RandAugment-style policy: each bag independently and
uniformly draws ONE operation from {uniform-CutMix+noise, subspace-noise, identity} and applies it. The
hope is a SINGLE augmentation that is robust across all three backbones — the patch-level backbones benefit
from the CutMix draws, TITAN survives because a third of its bags get the titan-friendly subspace-noise and
another third are left clean, diluting CutMix's collapse.

    per bag:  op ~ Uniform{cutmix+noise, subspace_noise, identity}
      cutmix+noise : replace round(strength*N) patches w/ uniform same-grade donors, then noise sigma=0.05
      subspace_noise: add noise ONLY in the nuisance complement of span{centroid_g - global_mean}
      identity     : unchanged
    label preserved.

`strength` = CutMix fraction (used on cutmix draws). Requires the train pool. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SIGMA = 0.05
SUB_SIGMA = 0.08     # subspace-noise level (titan-helpful range)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._U: Optional[torch.Tensor] = None
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
        bank, cents = {}, []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            cents.append(allp.mean(dim=0, keepdim=True))
        self._bank = bank
        C = torch.cat(cents, dim=0)
        M = C - C.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(M, full_matrices=False)
            r = max(1, min(M.shape[0] - 1, Vh.shape[0]))
            self._U = Vh[:r].contiguous()
        except Exception:
            self._U = None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        op = int(torch.randint(3, (1,)).item())
        if op == 2:                                   # identity
            return features, float(label)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        if op == 0:                                   # uniform cutmix + noise
            g = int(round(float(label)))
            bank = self._bank.get(g)
            out = features.clone()
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out[dst] = bank[src].to(features.device, features.dtype)
            out = out + torch.randn_like(out) * (std * SIGMA)
            return out, float(label)
        # op == 1: subspace noise (nuisance complement)
        noise = torch.randn_like(features) * (std * SUB_SIGMA)
        if self._U is not None:
            U = self._U.to(features.device, features.dtype)
            noise = noise - (noise @ U.t()) @ U
        return features + noise, float(label)
