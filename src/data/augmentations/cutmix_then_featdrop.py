"""cutmix_then_featdrop - uniform within-grade CutMix THEN feature-channel dropout (train-only).

cutmix_then_noise (uniform CutMix + additive Gaussian noise) is the ASGAP-beating augmentation. This swaps
the additive-noise regulariser for multiplicative feature-CHANNEL dropout: after the CutMix, a fraction P
of the feature dimensions are zeroed (independently per patch) and survivors rescaled by 1/(1-P). Dropout
is a structurally different regulariser (sparsifying, scale-preserving in expectation) than Gaussian jitter
and may suit the sparse-attention ASGAP head differently. Distinct from cutmix_then_dropout, which drops
whole PATCHES; this drops feature CHANNELS. Uniform donors keep the ASGAP-preferred diversity.

    1. replace round(strength*N) patches with UNIFORM same-grade donors
    2. m ~ Bernoulli(1-P) elementwise over [N, D] ;  out <- out * m / (1-P)          label preserved.

`strength` = cutmix fraction. P = channel dropout rate (module constant). Requires the train pool. MPS-safe.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

P = 0.1     # feature-channel dropout rate


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
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        keep = 1.0 - P
        m = (torch.rand_like(out) < keep).to(out.dtype)
        out = out * m / keep
        return out, float(label)
