"""subspace_mixup - signal-preserving cross-patient MixUp (train-only). Synthesis of the two winners.

The two augmentations that worked this study do so for different reasons:
  - cutmix injects REAL same-grade patches from other patients (genuine cross-patient diversity) but
    REPLACES patches wholesale -> collapses slide-level titan.
  - subspace_noise perturbs only the nuisance subspace (grade signal preserved) -> the only thing that
    helps titan, but its noise is synthetic (no real cross-patient information).

subspace_mixup combines both: move each patch a fraction of the way toward a REAL same-grade donor
patch, but ONLY along the nuisance directions (orthogonal to the grade-signal subspace span{c_g-mean}).
This injects real cross-patient nuisance variation WITHOUT touching the grade signal and WITHOUT the
destructive wholesale replacement -> it should help titan (per-patch, mild, signal-preserving) as well
as the patch-level backbones.

  delta = donor_same_grade - h ;  delta_perp = delta - proj_signal(delta) ;  h' = h + strength * delta_perp

`strength` = interpolation fraction toward the real donor (in nuisance directions). Label preserved.
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
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

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None or self._U is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)          # [r, D]
        n = features.shape[0]
        src = torch.randint(bank.shape[0], (n,))                 # one real same-grade donor per patch
        donor = bank[src].to(features.device, features.dtype)    # [N, D]
        delta = donor - features                                 # toward the real donor
        delta_perp = delta - (delta @ U.t()) @ U                 # keep only nuisance directions
        return features + self.s * delta_perp, float(label)
