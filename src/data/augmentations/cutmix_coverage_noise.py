"""cutmix_coverage_noise - CutMix with coverage-gap-filling donors + noise (train-only).

Active-diversity variant: for each bag, sample a candidate set of same-grade patches and pick as donors the
ones FARTHEST (max min-distance) from the current bag's patches, i.e. the donors that best fill the bag's
under-covered regions of the grade manifold. Distinct from FPS (global max-diversity, which over-diversified
and hurt) because coverage is measured RELATIVE to the current bag. Then all-patch noise. `strength` =
cutmix fraction. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
N_CAND = 256     # candidate donors sampled per bag before coverage selection


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 8000):
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
        nc = min(N_CAND, bank.shape[0])
        cand_idx = torch.randperm(bank.shape[0])[:nc]
        cand = bank[cand_idx]                                  # [nc, D]
        d = torch.cdist(cand, features)                        # [nc, n]
        mind = d.min(dim=1).values                             # coverage: dist to nearest bag patch
        sel = torch.topk(mind, min(k, nc), largest=True).indices
        donors = cand[sel]
        if donors.shape[0] < k:                                # pad if needed
            extra = cand[torch.randint(nc, (k - donors.shape[0],))]
            donors = torch.cat([donors, extra], dim=0)
        dst = torch.randperm(n)[:k]
        out = features.clone()
        out[dst] = donors[:k]
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
