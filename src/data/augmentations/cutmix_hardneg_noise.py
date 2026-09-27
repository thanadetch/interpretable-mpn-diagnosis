"""cutmix_hardneg_noise - same-grade CutMix + a few ADJACENT-grade hard-negative patches + noise (train-only).

Motivated by the sparse-attention (ASGAP/entmax) mechanism: entmax must actively zero out non-diagnostic
tissue. This injects a few HARD NEGATIVES — patches drawn from an ADJACENT grade (g-1 or g+1) — into the bag
while KEEPING the true hard label. The model must learn to attend to the true-grade evidence and suppress
the confusable adjacent-grade patches, sharpening the decision boundary. Distinct from cutmix_ordinal, which
SOFTENED the regression target toward the adjacent grade; here the label is unchanged (these are label
noise / hard negatives, not interpolation). The bulk of the bag is the winning same-grade uniform CutMix.

    1. replace round(strength*N) patches with UNIFORM same-grade donors
    2. replace round(BETA*N) further patches with UNIFORM adjacent-grade donors (label kept)
    3. add all-patch Gaussian noise sigma=0.05                                     label preserved.

`strength` = same-grade cutmix fraction. BETA = hard-negative fraction. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
BETA = 0.10     # fraction of hard-negative (adjacent-grade) patches


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
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank
        self._grades = sorted(bank.keys())

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
        perm = torch.randperm(n)
        k = min(max(1, int(round(self.s * n))), n)
        dst = perm[:k]
        src = torch.randint(bank.shape[0], (k,))
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        # hard negatives from an adjacent grade
        adj = [h for h in (g - 1, g + 1) if h in self._bank and self._bank[h].shape[0] > 0]
        kb = int(round(BETA * n))
        if adj and kb > 0:
            hb = adj[int(torch.randint(len(adj), (1,)).item())]
            abank = self._bank[hb]
            hdst = perm[k:k + kb]
            if hdst.numel() > 0:
                hsrc = torch.randint(abank.shape[0], (hdst.numel(),))
                out[hdst] = abank[hsrc].to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
