"""signal_stretch_noise - amplify the grade-signal component + add nuisance noise (train-only, titan-targeted).

subspace_noise (nuisance-only noise) is the ONLY augmentation that helps slide-level TITAN, but it only
regularises — it never sharpens the grade signal. This variant adds a second, complementary move: it
mildly AMPLIFIES each patch's projection onto the grade-signal subspace U = span{centroid_g - global_mean}
(stretching patches along the directions that separate grades, increasing inter-grade separability) WHILE
adding noise only in the orthogonal nuisance complement. No CutMix -> titan-safe. The hope is to push the
titan cell past subspace_noise's ceiling (0.804/0.9639) by making the grade signal more salient to the
slide-level aggregator.

    sig  = (h @ U^T) @ U                       # grade-signal component
    nz   = noise - (noise @ U^T) @ U           # nuisance noise (orthogonal to U)
    h'   = h + BETA*sig + nz                   # amplify signal, jitter nuisance   label preserved.

`strength` = nuisance noise level (sigma). BETA = signal amplification. Requires the train pool. MPS-safe.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.08)
BETA = 0.15     # grade-signal amplification factor


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.08, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._U: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            chunks[g].append(item[0].float())
        cents = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cents.append(allp.mean(dim=0, keepdim=True))
        C = torch.cat(cents, dim=0)
        M = C - C.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(M, full_matrices=False)
            r = max(1, min(M.shape[0] - 1, Vh.shape[0]))
            self._U = Vh[:r].contiguous()
        except Exception:
            self._U = None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._U is None:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        sig = (features @ U.t()) @ U                       # signal component
        noise = torch.randn_like(features) * (std * self.s)
        nz = noise - (noise @ U.t()) @ U                   # nuisance noise
        return features + BETA * sig + nz, float(label)
