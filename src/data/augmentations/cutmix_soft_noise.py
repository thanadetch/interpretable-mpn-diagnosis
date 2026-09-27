"""cutmix_soft_noise - SOFT (blended) uniform CutMix + feature noise (train-only).

Structural axis: discrete vs soft replacement. cutmix_then_noise fully replaces a patch with a donor; this
partially blends each selected patch toward its donor, h' = (1-lam)*h + lam*donor with lam ~ U(LAM_LO, 1),
a continuum between MixUp (soft) and CutMix (hard). Softening keeps a trace of the original patch, which
may be gentler on the bag geometry while still injecting donor diversity. Uniform donors (ASGAP-preferred),
then the usual all-patch noise. `strength` = fraction of patches blended. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
LAM_LO = 0.6     # minimum blend weight toward the donor


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
        lam = torch.empty(k, 1, device=features.device, dtype=features.dtype).uniform_(LAM_LO, 1.0)
        out = features.clone()
        out[dst] = (1.0 - lam) * out[dst] + lam * donors
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
