"""cutmix_featblock_noise - contiguous feature-BLOCK CutMix + noise (train-only).

Granularity axis: whole-patch swap (cutmix_then_noise) vs soft blend (cutmix_soft_noise) vs this — the true
feature-space analog of spatial CutMix, which cuts a contiguous REGION. Here, for the selected patches, a
contiguous block of BLOCK_FRAC*D feature channels is replaced by the donor's SAME channels (the rest of the
patch keeps its own values). Tests whether sub-patch (channel-block) mixing helps or, like random dim-swaps
(cutmix_nuisance_swap, which was catastrophic), breaks the coherent embedding. Uniform donors, then noise.
`strength` = fraction of patches block-mixed. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
BLOCK_FRAC = 0.5     # fraction of the D channels in the swapped contiguous block


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
        n, dsz = features.shape
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        donors = bank[src].to(features.device, features.dtype)
        bw = max(1, int(round(BLOCK_FRAC * dsz)))
        start = int(torch.randint(0, dsz - bw + 1, (1,)).item())
        out = features.clone()
        patch = out[dst].clone()
        patch[:, start:start + bw] = donors[:, start:start + bw]
        out[dst] = patch
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
