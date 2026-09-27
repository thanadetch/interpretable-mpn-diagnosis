"""cutmix_sizejitter_noise - champion CutMix + random BAG CARDINALITY + noise (train-only).

At strength 1.00 the champion recipe replaces every patch, i.e. the training bag is just N i.i.d. draws
from the grade's patch bank - and uni2/ASGAP scores the same there as at 0.7-0.9 (0.8281/0.9571), so bag
IDENTITY carries no information. Together with the 2026-07-25 finding that the stage-2 perturbation is
interchangeable, the training distribution has only two free parameters left: WHICH patches (destination
rule -> cutmix_adjclean_noise) and HOW MANY patches per bag - the one this module varies.

Bag size is never augmented anywhere in the registry: every module returns exactly N patches (patch_dropout
only shrinks, and it shrinks REAL bags, which was destructive). Here the bag is already ~80% synthetic, so
cardinality can be jittered both ways cheaply:

    1. within-grade CutMix @ 0.8 (the champion's uniform-donor stage)
    2. resize the bag to M = round(N * u), u ~ U(1-strength, 1+strength):
       M < N -> random subsample;  M > N -> append (M-N) extra uniform same-grade donors
    3. feature_noise @ 0.05

Rationale: attention pooling normalises over N, but the sparse (entmax) head's support size and the
softmax temperature interact with N, so ROI-to-ROI size variation is a real nuisance the model currently
never sees perturbed. `strength` = size-jitter half-range (0.3 -> bags of 0.7N..1.3N). MPS-safe,
deterministic, no trainer edits.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)

CUTMIX = 0.8
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3, max_bank: int = 20000):
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
            chunks[int(round(float(item[1])))].append(item[0].float())
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        g = int(round(float(label)))
        bank = self._bank.get(g) if self._bank is not None else None
        out = features
        n = features.shape[0]
        if bank is not None and bank.shape[0] >= 1:
            # --- step 1: uniform within-grade cutmix @ CUTMIX ---
            k = min(max(1, int(round(CUTMIX * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
            # --- step 2: random bag cardinality ---
            if self.s > 0.0:
                lo, hi = max(0.05, 1.0 - self.s), 1.0 + self.s
                u = float(torch.rand(1).item()) * (hi - lo) + lo
                m = max(4, int(round(u * n)))
                if m < n:
                    out = out[torch.randperm(n)[:m]]
                elif m > n:
                    extra = bank[torch.randint(bank.shape[0], (m - n,))].to(features.device, features.dtype)
                    out = torch.cat([out, extra], dim=0)
        # --- step 3: feature_noise @ 0.05 ---
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
