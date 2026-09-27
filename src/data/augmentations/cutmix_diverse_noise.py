"""cutmix_diverse_noise - MAXIMALLY-diverse within-grade CutMix donors + feature noise (train-only).

Directly motivated by this study's finding that the sparse-attention ASGAP aggregator wins with DIVERSE
(uniform) donors and is hurt by concentrated prototype donors (soft ABMIL is the opposite). If diversity is
what helps ASGAP, push it to the limit: build a maximally-spread donor subset per grade via farthest-point
sampling (FPS), then draw donors uniformly from that spread subset. FPS greedily picks patches that are as
far as possible from those already picked, so the donor pool covers the grade's full within-class variation
rather than clustering near typical tissue. A small global feature-noise step follows (the ASGAP-helping
regulariser from cutmix_then_noise).

    _ensure:  per grade, FPS a spread subset of SUBSET points from the bank.
    per bag:  replace round(strength*N) patches with donors drawn uniformly from the spread subset;
              then add N(0, (per-dim std * SIGMA)^2) to every patch.   label preserved.

`strength` = cutmix fraction. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SUBSET = 256     # size of the FPS-selected diverse donor pool per grade
SIGMA = 0.05


def _fps(x: torch.Tensor, m: int) -> torch.Tensor:
    """Greedy farthest-point sampling: return indices of m maximally-spread rows of x [N, D]."""
    n = x.shape[0]
    m = min(m, n)
    idx = torch.empty(m, dtype=torch.long)
    idx[0] = torch.randint(n, (1,)).item()
    d = torch.norm(x - x[idx[0]].unsqueeze(0), dim=1)
    for i in range(1, m):
        idx[i] = int(torch.argmax(d).item())
        d = torch.minimum(d, torch.norm(x - x[idx[i]].unsqueeze(0), dim=1))
    return idx


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 8000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._sub: Optional[Dict[int, torch.Tensor]] = None
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
        sub = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            idx = _fps(allp, SUBSET)
            sub[g] = allp[idx].contiguous()      # [<=SUBSET, D] maximally-spread donor pool
        self._sub = sub

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._sub is None:
            return features, float(label)
        g = int(round(float(label)))
        sub = self._sub.get(g)
        if sub is None or sub.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(sub.shape[0], (k,))
        out = features.clone()
        out[dst] = sub[src].to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
