"""cutmix_proto_prob - champion prototype CutMix applied with a Bernoulli probability (train-only).

proto_stoch varies the cutmix FRACTION per bag but always augments (fraction in [0.7,0.9], never 0).
This variant instead keeps the champion's fixed aggressive fraction but applies it only to a RANDOM
subset of bags each epoch: with probability P a bag is cut-mixed at `strength`, otherwise it is passed
through fully clean. The training distribution is therefore BIMODAL — a mix of pristine bags (which
preserve real grade evidence and support val) and strongly-regularised bags (which support test) — the
standard "augmentation probability" lever, orthogonal to the strength sweep and never tried here.

    per bag:  with prob P -> prototype CutMix at fraction `strength`;  else -> unchanged.

`strength` = cutmix fraction when applied. P = application probability (module constant). Requires the
train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
P = 0.6     # probability a bag is augmented


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
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
        bank, weights = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
        self._bank, self._w = bank, weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        if float(torch.rand(1).item()) >= P:      # skip -> clean bag
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
