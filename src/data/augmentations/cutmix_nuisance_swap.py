"""cutmix_nuisance_swap - transplant only the NUISANCE (style) component of donor patches (train-only).

Combines this study's two positive levers:
  - the grade-signal subspace U = orthonormal span{centroid_g - global_mean} (from subspace_noise, the only
    aug that helped TITAN), which isolates the directions that carry grade information, and
  - prototype-weighted same-grade CutMix (the virchow2 champion).

Instead of replacing whole patches (which swaps signal AND style), for each selected patch we KEEP its own
grade-signal component (projection onto U) and replace only its NUISANCE component (the orthogonal
complement = texture / stain / scanner style) with a near-centroid donor's nuisance. The bag thus keeps its
real per-patch grade evidence while its style is re-mixed within the grade — a signal-preserving style
augmentation, distinct from every whole-patch CutMix variant tried so far.

    signal(h) = (h @ U^T) @ U ;  nuisance(h) = h - signal(h)
    h' = signal(h) + nuisance(donor)            for round(strength*N) patches.  Label preserved.

Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4


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
        bank, weights, cents = {}, {}, []
        all_for_mean = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            cents.append(c)
            all_for_mean.append(allp.mean(dim=0, keepdim=True))
        # signal subspace = span{centroid_g - global_mean}, orthonormalised via SVD
        C = torch.cat(cents, dim=0)                       # [G, D]
        gmean = C.mean(dim=0, keepdim=True)
        M = C - gmean                                     # [G, D], rank <= G-1
        try:
            _, _, Vh = torch.linalg.svd(M, full_matrices=False)
            r = max(1, min(M.shape[0] - 1, Vh.shape[0]))
            self._U = Vh[:r].contiguous()                 # [r, D] orthonormal signal basis
        except Exception:
            self._U = None
        self._bank, self._w = bank, weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None or self._U is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)   # [r, D]
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        donors = bank[src].to(features.device, features.dtype)      # [k, D]
        target = features[dst]                                      # [k, D]
        # signal(target) + nuisance(donor) = target - proj_U(target) removed?  keep signal(target):
        sig_target = (target @ U.t()) @ U                          # signal of kept patch
        nui_donor = donors - (donors @ U.t()) @ U                  # nuisance of donor
        out = features.clone()
        out[dst] = sig_target + nui_donor
        return out, float(label)
