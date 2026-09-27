"""cutmix_proto_pull - prototype CutMix whose donors are pulled toward the grade centroid (train-only).

The study champion cutmix_prototype_sharp already samples donor patches NEAR the grade centroid (TAU=0.4
softmax on distance). The bag_mix_samegrade negative result showed the opposite direction — donors carrying
raw intra-slide context — is worse, i.e. cleaner grade signal beats faithful context. This variant pushes
that winning lever one step further: after sampling a near-centroid donor, blend it a fraction BETA toward
the grade centroid, so the transplanted patches carry an even purer grade signal while the majority of
original patches (and their within-bag variance) are untouched.

    donor* = (1-beta)*donor + beta*centroid_g ,  then CutMix round(strength*N) patches as usual.

Distinct from prototype_shrink (which shrank EVERY patch toward the centroid = collapsed within-grade
variance, negative): here only the minority of REPLACEMENT patches are pulled; the kept patches keep full
variance. `strength` = cutmix fraction. BETA = donor pull (module constant). Requires the train pool.
Label preserved. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
BETA = 0.5   # fraction each donor is pulled toward the grade centroid


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
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        donors = bank[src].to(features.device, features.dtype)
        cc = c.to(features.device, features.dtype)
        donors = (1.0 - BETA) * donors + BETA * cc
        out = features.clone()
        out[dst] = donors
        return out, float(label)
