"""cutmix_then_bagshift - champion CutMix + a bag-COHERENT nuisance offset (train-only).

The champion ``cutmix_then_noise`` = uniform within-grade CutMix @ strength, then i.i.d. per-patch
Gaussian noise @ 0.05. Its second stage is the part that mostly cancels under pooling (see
``bag_nuisance_shift``). This module keeps stage 1 identical and REPLACES stage 2 with one shared
nuisance-subspace offset per bag - an acquisition-level shift that survives pooling - while leaving the
grade-signal subspace untouched.

  1. within-grade CutMix @ ``strength`` (uniform donors over patches = the established optimum)
  2. + v, one vector per bag, v ~ N(0, (SHIFT * sigma_bag)^2) projected off the grade-signal subspace,
    where sigma_bag = per-dim std of the training bag means (the real between-ROI nuisance scale)

If this beats cutmix_then_noise, the augmentation gain lives in perturbing the POOLED representation,
not the individual patches - a directly thesis-usable mechanism statement about MIL augmentation.

`strength` = cutmix fraction (the champion's 0.8 default); SHIFT below = offset size. MPS-safe,
deterministic given the global seed, no trainer edits.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SHIFT = 0.3


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._U: Optional[torch.Tensor] = None
        self._sigma: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        means = []
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            chunks[int(round(float(item[1])))].append(feats)
            means.append(feats.mean(dim=0))
        if not chunks:
            return
        bank: Dict[int, torch.Tensor] = {}
        cents = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            cents.append(allp.mean(dim=0))
        self._bank = bank
        if len(means) >= 2:
            M = torch.stack(means, dim=0)
            self._sigma = M.std(dim=0, keepdim=True).clamp(min=1e-6)
        if len(cents) >= 2:
            C = torch.stack(cents, dim=0)
            Md = C - C.mean(dim=0, keepdim=True)
            _, S, Vh = torch.linalg.svd(Md, full_matrices=False)
            rank = int((S > 1e-6 * S.max()).sum().item())
            self._U = Vh[:rank].contiguous()

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        out = features
        # --- step 1: cutmix_within_grade (uniform donors) ---
        if self.s > 0.0 and self._bank is not None:
            bank = self._bank.get(int(round(float(label))))
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        # --- step 2: one shared nuisance-subspace offset for the whole bag ---
        if self._U is not None and self._sigma is not None:
            U = self._U.to(out.device, out.dtype)
            sig = self._sigma.to(out.device, out.dtype)
            v = torch.randn_like(sig) * (sig * SHIFT)
            v = v - (v @ U.t()) @ U
            out = out + v
        return out, float(label)
