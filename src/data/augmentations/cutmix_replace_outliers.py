"""cutmix_replace_outliers - prototype CutMix that replaces the LEAST grade-typical patches (train-only).

The champion cutmix_prototype_sharp replaces RANDOM patches with near-centroid donors. Here we replace the
patches FARTHEST from the grade centroid (the least grade-typical / most ambiguous tissue in the bag) and
KEEP the near-centroid confident originals. The bag therefore trades its ambiguous evidence for clean
prototype signal while preserving the patches that already vote confidently for the grade — a targeted swap
that should raise grade purity without discarding the bag's strongest real evidence.

    per bag:  dst = the round(strength*N) patches with LARGEST distance to centroid_g;
              replace them with near-centroid (TAU=0.4) prototype donors.  Label preserved.

Distinct from cutmix_worst_preserve (which ranked by the fibrosis severity axis c_G3-c_G0): this ranks by
distance to the OWN-grade centroid, i.e. grade-typicality, symmetric across all grades. Requires the train
pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._cent: Optional[Dict[int, torch.Tensor]] = None
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
        bank, weights, cent = {}, {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            cent[g] = c
        self._bank, self._w, self._cent = bank, weights, cent

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g); c = self._cent.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        cc = c.to(features.device, features.dtype)
        d_own = torch.norm(features - cc, dim=1)          # distance of each bag patch to grade centroid
        dst = torch.topk(d_own, k, largest=True).indices  # the least grade-typical patches
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
