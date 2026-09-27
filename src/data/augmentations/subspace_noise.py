"""subspace_noise - anisotropic feature noise in the NUISANCE subspace only (train-only).

Isotropic feature_noise (h += std*N(0,1)) was a fluke/negative because it corrupts the grade-signal
directions as much as the nuisance ones. This augmentation is principled: add noise ONLY in the subspace
ORTHOGONAL to the grade-discriminative directions, so it perturbs patient/stain/sampling nuisance
variation while leaving the grade signal intact.

  1. grade-signal subspace = span{ centroid_g - global_mean : g }  (rank <= #grades-1 = 3), orthonormalised.
  2. per patch: noise ~ N(0, (within-bag per-dim std * strength)^2); remove its component in the signal
     subspace; add only the orthogonal (nuisance) part.  Label preserved.

Tests the hypothesis that noise-augmentation fails only because it hits the signal — protect the signal
and it should regularise. `strength` = noise magnitude (fraction of per-dim std). MPS-safe, deterministic.
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
        self._U: Optional[torch.Tensor] = None   # [r, D] orthonormal signal-subspace basis
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
        cents = []
        allmeans = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0)
            cents.append(c)
            allmeans.append((c, allp.shape[0]))
        if len(cents) < 2:
            return
        C = torch.stack(cents, dim=0)                    # [G, D]
        gmean = C.mean(dim=0, keepdim=True)
        M = C - gmean                                    # [G, D] centred class directions
        # orthonormal basis of row-space of M via SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)  # Vh: [G, D]
        rank = int((S > 1e-6 * S.max()).sum().item())
        self._U = Vh[:rank].contiguous()                 # [r, D] rows = orthonormal signal directions

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._U is None:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)          # [r, D]
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)  # [1, D]
        noise = torch.randn_like(features) * (std * self.s)      # [N, D]
        proj = (noise @ U.t()) @ U                               # component in signal subspace
        noise_perp = noise - proj                                # nuisance-only noise
        return features + noise_perp, float(label)
