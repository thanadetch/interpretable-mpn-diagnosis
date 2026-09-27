"""cutmix_bankboot_noise - per-EPOCH resampled donor bank (bagging over donors) + noise (train-only).

Third module on the TIME axis, and the only one that changes WHAT is available rather than HOW MUCH is
applied. Every CutMix variant in the registry draws donors from the full grade bank at every step, so over
an epoch the donor distribution is exactly the grade's empirical patch distribution - identical every
epoch. Here the bank is re-drawn each epoch:

    at the start of each epoch, keep a random BOOT_FRAC subset of each grade's patch bank; all donors of
    that epoch come from that subset (then CutMix @ ``strength``, then feature_noise @ 0.05)

so different epochs see different donor "worlds" - bagging/bootstrap applied to the augmentation source
rather than to the bags. The gradient signal decorrelates across epochs while each epoch stays internally
consistent, which is what makes bootstrap ensembling work; it also stress-tests the finding that donor
diversity is what ASGAP wants: a per-epoch restricted bank reduces within-epoch diversity but increases
across-epoch diversity, separating the two for the first time.

Epoch is derived from the call count (epoch = calls // len(train_pool)); no trainer signal is needed.
`strength` = cutmix fraction; BOOT_FRAC = share of each bank kept per epoch. MPS-safe, deterministic
given the global seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

BOOT_FRAC = 0.5
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._epoch_bank: Dict[int, torch.Tensor] = {}
        self._epoch = -1
        self._n = 1
        self._calls = 0
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        self._n = max(1, len(pool))
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            chunks[int(round(float(item[1])))].append(item[0].float())
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def _refresh(self, epoch: int) -> None:
        if epoch == self._epoch or self._bank is None:
            return
        self._epoch = epoch
        eb = {}
        for g, bank in self._bank.items():
            m = max(1, int(round(BOOT_FRAC * bank.shape[0])))
            eb[g] = bank[torch.randperm(bank.shape[0])[:m]]
        self._epoch_bank = eb

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        self._refresh(self._calls // self._n)
        self._calls += 1
        out = features
        if self.s > 0.0:
            bank = self._epoch_bank.get(int(round(float(label))))
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
