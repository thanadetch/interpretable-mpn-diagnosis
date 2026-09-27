"""cutmix_ordinal - ordinal cross-grade CutMix with interpolated regression target (train-only).

All prior cutmix variants transplant SAME-grade patches and keep the label fixed. This one exploits
the *ordinal / regression* nature of fibrosis grading: a grade is roughly monotonic in the density of
reticulin fibrosis, so a bag that is a mixture of grade-g and grade-(g±1) tissue should carry an
INTERMEDIATE severity. We therefore synthesise on-manifold "between-grade" bags to densify the ordinal
continuum that a 4-class cohort samples only coarsely.

    pick a neighbour grade gd in {g-1, g+1};  replace a fraction p of the bag's patches with real
    grade-gd patches drawn near the gd prototype (softmax(-dist/centroid), like cutmix_prototype_sharp);
    set the regression target to the mixing-interpolated severity:  y = (1-p)*g + p*gd = g + p*(gd-g).

`strength` = p = mixed fraction = magnitude of the target shift toward the neighbour grade.
Real patches only (on-manifold). Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)

TAU = 0.4  # softmax temperature on distance-to-centroid (matches cutmix_prototype_sharp)


class Augmentation(BaseAugmentation):
    requires_regression = True  # produces fractional targets -> regression only

    def __init__(self, strength: float = 0.3, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._grades: Optional[List[int]] = None
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
            centroid = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - centroid, dim=1)
            w = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            weights[g] = w
        self._bank = bank
        self._w = weights
        self._grades = sorted(bank.keys())

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        # candidate neighbour grades that actually exist in the bank
        cands = [gd for gd in (g - 1, g + 1) if gd in self._bank and self._bank[gd].shape[0] >= 1]
        if not cands:
            return features, float(label)
        gd = cands[int(torch.randint(len(cands), (1,)).item())]
        w = self._w[gd]; bank = self._bank[gd]
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n - 1)   # keep at least 1 original patch
        p = k / n                                        # realised mixing fraction
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)  # prototype-weighted neighbour donors
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        target = (1.0 - p) * g + p * gd                  # interpolated ordinal severity
        return out, float(target)
