"""cutmix_uni_subspace_n05 - compose the two winners with the ASGAP-correct donor scheme (train-only).

The existing cutmix_then_subspace uses PROTOTYPE (centroid-weighted) donors, which per the diversity finding
crashes ASGAP test (entmax wants uniform/diverse donors). This variant swaps step 1 to UNIFORM within-grade
CutMix — the same donor scheme as the champion cutmix_then_noise — and keeps step 2 = signal-preserving
nuisance-subspace noise (the titan-ABMIL win). Goal: a single aug that stacks the virchow2 cutmix win and the
titan nuisance-noise win WITHOUT the prototype penalty on ASGAP.

  step 1 = uniform within-grade CutMix @ `strength`  (replace a fraction of patches with random same-grade donors)
  step 2 = noise ONLY in span orthogonal to grade-signal directions span{c_g - mean}, scaled by bag std * NOISE

`strength` = cutmix fraction; noise level fixed at NOISE. Label preserved. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

NOISE = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._U: Optional[torch.Tensor] = None
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
        cents = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            cents.append(allp.mean(dim=0))
        self._bank = bank
        if len(cents) >= 2:
            C = torch.stack(cents, dim=0)
            M = C - C.mean(dim=0, keepdim=True)
            _, S, Vh = torch.linalg.svd(M, full_matrices=False)
            rank = int((S > 1e-6 * S.max()).sum().item())
            self._U = Vh[:rank].contiguous()

    @torch.no_grad()
    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        out = features
        # step 1: uniform within-grade cutmix (ASGAP-friendly donor scheme)
        bank = self._bank.get(g)
        if bank is not None and bank.shape[0] >= 1:
            n = features.shape[0]
            k = min(max(1, int(round(self.s * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        # step 2: signal-preserving nuisance noise
        if self._U is not None:
            U = self._U.to(features.device, features.dtype)
            std = out.std(dim=0, keepdim=True).clamp(min=1e-6)
            noise = torch.randn_like(out) * (std * NOISE)
            out = out + (noise - (noise @ U.t()) @ U)
        return out, float(label)
