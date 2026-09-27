"""cutmix_covnoise - uniform CutMix + COVARIANCE-shaped (colored) noise (train-only).

All prior noise was diagonal (independent per-dim std). This shapes the noise to the grade's covariance:
noise ~ N(0, sigma^2 * Sigma_g) built from the top-R principal components of the grade bank (V * sqrt(S)).
Colored noise perturbs mostly along the grade's high-variance directions (the natural within-grade
variation) instead of equally per axis, which may be a more on-manifold regulariser. Uniform donors keep
the ASGAP-preferred diversity. `strength` = cutmix fraction. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
R = 64        # principal components retained for the noise covariance


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._basis: Optional[Dict[int, torch.Tensor]] = None    # V*sqrt(S)/sqrt(N) per grade [R,D]
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
        bank, basis = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            c = allp - allp.mean(dim=0, keepdim=True)
            try:
                _, S, Vh = torch.linalg.svd(c, full_matrices=False)
                r = int(min(R, Vh.shape[0]))
                scale = (S[:r] / (c.shape[0] ** 0.5)).unsqueeze(1)     # sqrt(eigenvalue) [r,1]
                basis[g] = Vh[:r] * scale                              # [r, D]
            except Exception:
                basis[g] = None
        self._bank, self._basis = bank, basis

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); basis = self._basis.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        if basis is not None:
            B = basis.to(features.device, features.dtype)             # [r, D]
            coeff = torch.randn(n, B.shape[0], device=features.device, dtype=features.dtype)
            out = out + SIGMA * (coeff @ B)                           # colored noise ~ N(0, sigma^2 Sigma)
        return out, float(label)
