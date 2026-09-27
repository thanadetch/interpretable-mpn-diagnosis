"""cutmix_uni_signal - CONTROL for cutmix_uni_subspace: noise in the SIGNAL subspace (train-only).

Identical to cutmix_uni_subspace except step 2 keeps ONLY the grade-signal component of the noise instead of
removing it: out += (noise @ U.t()) @ U, i.e. noise is projected ONTO span{c_g - mean} rather than orthogonal
to it. This is the mechanistic control: if perturbing the grade-signal directions HURTS while perturbing the
nuisance directions (cutmix_uni_subspace) HELPS on titan/ASGAP, that pins the win to the signal-preserving
property of the nuisance-subspace noise. `strength` = cutmix fraction; noise level fixed at NOISE. Label
preserved. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

NOISE = 0.1


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
        # step 1: uniform within-grade cutmix (same as cutmix_uni_subspace)
        bank = self._bank.get(g)
        if bank is not None and bank.shape[0] >= 1:
            n = features.shape[0]
            k = min(max(1, int(round(self.s * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        # step 2 (CONTROL): keep ONLY the signal-subspace component of the noise
        if self._U is not None:
            U = self._U.to(features.device, features.dtype)
            std = out.std(dim=0, keepdim=True).clamp(min=1e-6)
            noise = torch.randn_like(out) * (std * NOISE)
            out = out + (noise @ U.t()) @ U
        return out, float(label)
