"""cutmix_adjblur_noise - within-grade CutMix + small adjacent-grade boundary-blur + noise (train-only).

Champion cutmix_then_noise replaces patches only with SAME-grade donors. This adds a small boundary-blur term:
after the main within-grade transplant (fraction `strength`), an extra small fraction BLUR of patches is
replaced with donors from an ADJACENT grade (g-1 or g+1, chosen toward the interior of the 0..3 range), but
the regression target is kept HARD at g. Rationale: the main same-grade cutmix preserves grade signal (why the
champion works); a tiny dose of next-grade texture with the label unchanged teaches tolerance around the
decision boundary without moving the target (distinct from cutmix_ordinal, which interpolated the target and
lost). Then the champion's isotropic all-patch noise. `strength` = same-grade cutmix fraction. MPS-safe,
deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
BLUR = 0.1  # fraction of patches replaced with adjacent-grade donors (label kept hard)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._grades = None
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
        self._grades = sorted(bank.keys())

    def _adjacent(self, g: int) -> Optional[int]:
        # prefer the neighbour toward the interior; fall back to whichever exists
        gmin, gmax = self._grades[0], self._grades[-1]
        cands = [h for h in (g + 1, g - 1) if h in self._bank]
        if not cands:
            return None
        # bias toward interior (away from range ends)
        cands.sort(key=lambda h: abs(h - (gmin + gmax) / 2.0))
        return cands[0]

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
        out = features.clone()
        # main same-grade cutmix
        k = min(max(1, int(round(self.s * n))), n)
        perm = torch.randperm(n)
        dst = perm[:k]
        src = torch.randint(bank.shape[0], (k,))
        out[dst] = bank[src].to(features.device, features.dtype)
        # small adjacent-grade boundary blur (label kept hard)
        h = self._adjacent(g)
        if h is not None:
            adjbank = self._bank[h]
            kb = min(max(1, int(round(BLUR * n))), n - k) if k < n else 0
            if kb > 0:
                dstb = perm[k:k + kb]
                srcb = torch.randint(adjbank.shape[0], (kb,))
                out[dstb] = adjbank[srcb].to(features.device, features.dtype)
        # isotropic noise
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
