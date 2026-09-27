"""coral_subspace - SECOND-ORDER (covariance) statistics transfer between same-grade bags (train-only).

MixStyle (tested, on the frontier) mixes FIRST-order per-dimension statistics (mean and std) between bags.
Nothing in the registry has ever touched the SECOND-order structure - how the feature dimensions co-vary
inside a bag - which is where the "texture" of a bag's feature cloud lives: two ROIs of the same grade can
share means and per-dim spreads yet have completely different correlation structure (elongated vs
isotropic clouds, i.e. a field dominated by one fibre orientation/density mode vs a heterogeneous field).

Full CORAL is impossible here (D up to 2560, only ~40 patches per bag = rank-40 covariance), so the
transfer is done inside the grade bank's top-K principal subspace, where the covariance is estimable:

    1. K = top principal directions of the grade's patch bank (cached once, K = 16)
    2. project the recipient bag and a random same-grade donor bag into that subspace
    3. whiten the recipient's K x K covariance and re-colour it with the donor's (Cholesky), then
       interpolate: z' = (1 - s) * z + s * z_recoloured
    4. map back and add the untouched orthogonal residual, so only the second-order structure changes -
       the bag keeps its own mean and its own patches' identity outside the subspace

`strength` = interpolation weight toward the donor's covariance (0.5 default). Label preserved. No new
data, no trainer edits. MPS-safe (Cholesky on a 16x16 matrix, on CPU-side cached basis), deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

K = 16
EPS = 1e-4


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._basis: Dict[int, torch.Tensor] = {}        # grade -> [K, D]
        self._bags: Dict[int, List[torch.Tensor]] = defaultdict(list)
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            chunks[g].append(feats)
            self._bags[g].append(feats)
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            X = allp - allp.mean(dim=0, keepdim=True)
            try:
                _, _, Vh = torch.linalg.svd(X, full_matrices=False)
            except Exception:
                continue
            self._basis[g] = Vh[:K].contiguous()

    @staticmethod
    def _chol(cov: torch.Tensor) -> Optional[torch.Tensor]:
        k = cov.shape[0]
        reg = cov + EPS * torch.eye(k, dtype=cov.dtype) * cov.diagonal().mean().clamp(min=EPS)
        try:
            return torch.linalg.cholesky(reg)
        except Exception:
            return None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < K:
            return features, float(label)
        self._ensure(pool)
        g = int(round(float(label)))
        U = self._basis.get(g)
        bags = self._bags.get(g)
        if U is None or not bags:
            return features, float(label)
        donor = bags[int(torch.randint(len(bags), (1,)).item())]
        if donor.shape[0] < K or donor.shape[1] != features.shape[1]:
            return features, float(label)

        f = features.detach().to("cpu", torch.float32)
        Ucpu = U
        z = (f - f.mean(0, keepdim=True)) @ Ucpu.t()                     # [N, K]
        zd = (donor - donor.mean(0, keepdim=True)) @ Ucpu.t()            # [M, K]
        cs = z.t() @ z / max(1, z.shape[0] - 1)
        cd = zd.t() @ zd / max(1, zd.shape[0] - 1)
        Ls, Ld = self._chol(cs), self._chol(cd)
        if Ls is None or Ld is None:
            return features, float(label)
        try:
            w = torch.linalg.solve_triangular(Ls, z.t(), upper=False).t()  # whitened [N, K]
        except Exception:
            return features, float(label)
        z_re = w @ Ld.t()                                                 # re-coloured with donor cov
        z_new = (1.0 - self.s) * z + self.s * z_re
        delta = (z_new - z) @ Ucpu                                        # [N, D] change inside subspace
        return features + delta.to(features.device, features.dtype), float(label)
