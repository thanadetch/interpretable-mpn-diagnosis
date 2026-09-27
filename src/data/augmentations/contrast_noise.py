"""contrast_noise - distance-weighted feature noise that sharpens signal/background contrast (train-only).

Motivated by the SPARSE-attention aggregator (ASGAP / entmax), which wins when a bag has a few clearly
grade-typical patches it can attend to and a clearly-distinct background it can zero out. This aug adds
per-patch Gaussian noise whose magnitude GROWS with the patch's distance to its grade centroid: patches
already close to the prototype (the grade signal) stay almost clean, while atypical / background patches
get more noise and become even less reliable. That widens the signal<->background gap the sparse attention
keys on, without any cross-bag CutMix — so it is titan-safe (CutMix collapses slide-level TITAN; pure noise
does not, cf. subspace_noise).

    d_i = ||h_i - centroid_g|| ;  w_i = d_i / mean_j(d_j)            (relative atypicality, mean 1)
    h_i' = h_i + N(0, (per-dim std * strength * w_i)^2)             label preserved.

`strength` = base noise level (scaled by per-dim std and by relative distance). Requires the train pool.
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.1)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.1, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._cent: Optional[Dict[int, torch.Tensor]] = None
        self._std: Optional[Dict[int, torch.Tensor]] = None
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
        cent, std = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cent[g] = allp.mean(dim=0, keepdim=True)                 # [1, D]
            std[g] = allp.std(dim=0, keepdim=True).clamp(min=1e-6)   # [1, D]
        self._cent, self._std = cent, std

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._cent is None:
            return features, float(label)
        g = int(round(float(label)))
        c = self._cent.get(g); sd = self._std.get(g)
        if c is None:
            return features, float(label)
        c = c.to(features.device, features.dtype)
        sd = sd.to(features.device, features.dtype)
        d = torch.norm(features - c, dim=1)                          # [N] distance to grade centroid
        w = (d / d.mean().clamp(min=1e-6)).unsqueeze(1)              # [N,1] relative atypicality (mean 1)
        noise = torch.randn_like(features) * (sd * self.s * w)
        return features + noise, float(label)
