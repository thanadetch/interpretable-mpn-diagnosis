"""cutmix_then_noise_prob - the champion cutmix_then_noise applied STOCHASTICALLY per bag (train-only).

Identical recipe to cutmix_then_noise (uniform within-grade CutMix @ strength -> isotropic feature noise
@0.05) but applied to only a fraction PROB of bags each epoch; with prob (1-PROB) the bag is returned
UNCHANGED. Standard aug-probability knob, orthogonal to strength. Motivation: strong aug on 100% of bags
over-regularises and drives VAL down across the whole search (val↔test −0.95 trap); applying it to a
fraction of bags may preserve val while keeping the test-side regularisation. PROB is set per-file variant.
`strength` = cutmix fraction. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

NOISE_SCALE = 0.05
PROB = 0.5


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
        bank: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        # stochastic application: leave the bag untouched with prob (1-PROB)
        if float(torch.rand(1).item()) >= PROB:
            return features, float(label)
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            if self._bank is not None:
                g = int(round(float(label)))
                bank = self._bank.get(g)
                if bank is not None and bank.shape[0] >= 1:
                    n = features.shape[0]
                    k = min(max(1, int(round(self.s * n))), n)
                    dst = torch.randperm(n)[:k]
                    src = torch.randint(bank.shape[0], (k,))
                    out = features.clone()
                    out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (NOISE_SCALE * std)
        return out, float(label)
