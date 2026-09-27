"""cutmix_then_subspace - compose the two winners: prototype CutMix + signal-preserving nuisance noise.

Combines the session's two positive augmentations:
  step 1 = cutmix_prototype_sharp @ `strength`  (replace a fraction of the bag's patches with real
           same-grade donors near the grade prototype; injects cross-patient tissue diversity — the
           robust virchow2 win).
  step 2 = subspace_noise @ 0.1  (add feature noise ONLY in the subspace orthogonal to the grade-signal
           directions span{c_g-mean}; perturbs nuisance variation without touching the grade — the titan win).

Hypothesis: the two act on different axes (real inter-patient diversity vs signal-preserving nuisance
noise), so composing them may stack on patch-level backbones. `strength` = the cutmix fraction; the noise
level is fixed at 0.1. Label preserved. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
NOISE = 0.1


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
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
        weights: Dict[int, torch.Tensor] = {}
        cents = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0)
            dist = torch.norm(allp - c.unsqueeze(0), dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            cents.append(c)
        self._bank = bank
        self._w = weights
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
        # step 1: prototype cutmix
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is not None and bank.shape[0] >= 1:
            n = features.shape[0]
            k = min(max(1, int(round(self.s * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.multinomial(w, k, replacement=True)
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        # step 2: signal-preserving nuisance noise
        if self._U is not None:
            U = self._U.to(features.device, features.dtype)
            std = out.std(dim=0, keepdim=True).clamp(min=1e-6)
            noise = torch.randn_like(out) * (std * NOISE)
            out = out + (noise - (noise @ U.t()) @ U)
        return out, float(label)
