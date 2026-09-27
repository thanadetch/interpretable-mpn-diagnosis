"""cutmix_nn_noise - nearest-neighbour within-grade CutMix + feature noise (train-only).

Every prior CutMix drew donors randomly / by prototype / by FPS — all can transplant patches far from the
one they replace, which is what pushes slide-level TITAN off-manifold and collapses it. This variant instead
replaces each selected patch with one of ITS OWN k nearest neighbours in the same-grade bank: a "different
but locally similar" real patch. The transplant is maximally on-manifold (minimal disruption to the bag's
geometry / mean) yet still injects genuine cross-ROI variation, then a small feature-noise step regularises.

Hypotheses: (a) on-manifold NN donors may let TITAN tolerate CutMix (unlike random donors); (b) they give
ASGAP realistic within-grade diversity without outlier donors (which cutmix_diverse_noise showed hurt).

    _ensure: per grade, a (subsampled) bank.
    per bag: for each of round(strength*N) selected patches, donor = a random pick among its K nearest
             neighbours in the grade bank; replace; then add N(0,(per-dim std*SIGMA)^2) to every patch.
    label preserved.

`strength` = cutmix fraction. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

K_NN = 5        # pick donor among the K nearest neighbours (avoids exact-self, adds slight variety)
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 4000):
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
        if bank is None or bank.shape[0] < 2:
            return features, float(label)
        bank = bank.to(features.device, features.dtype)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        # k nearest neighbours of each selected patch in the grade bank
        q = features[dst]                                   # [k, D]
        d = torch.cdist(q, bank)                            # [k, B]
        knn = min(K_NN, bank.shape[0])
        nn_idx = torch.topk(d, knn, dim=1, largest=False).indices    # [k, knn]
        pick = torch.randint(knn, (k,), device=features.device)
        chosen = nn_idx[torch.arange(k, device=features.device), pick]   # [k]
        out = features.clone()
        out[dst] = bank[chosen]
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
