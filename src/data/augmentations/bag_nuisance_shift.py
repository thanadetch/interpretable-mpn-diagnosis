"""bag_nuisance_shift - ONE shared nuisance-subspace offset per bag (acquisition-shift model, train-only).

Every noise-type augmentation in the registry (feature_noise, subspace_noise, contrast_noise, all the
*_noise compositions) perturbs each patch INDEPENDENTLY. For an attention-MIL head that pools patches,
i.i.d. per-patch noise largely CANCELS in the pooled representation (it shrinks as 1/sqrt(N) with
N ~ hundreds of patches), so it barely perturbs the quantity the grade is actually read from. That is a
concrete reason the noise-only family was inert ([[session-negatives-2026-07-24]] #1).

This module perturbs the bag the way the real nuisance does: ONE displacement vector shared by ALL
patches of the bag - a coherent acquisition-level offset (stain batch, scanner, illumination, section
thickness), which is what actually differs between ROIs. It does NOT cancel under pooling.

  1. signal subspace U = span{ centroid_g - mean_of_centroids } (rank <= 3), orthonormalised - as in
     subspace_noise, so the grade signal is protected;
  2. sigma_bag = per-dim std of the training BAG MEANS = the observed between-ROI nuisance scale;
  3. per bag: v ~ N(0, (strength * sigma_bag)^2), remove its signal-subspace component, add the
     remaining nuisance offset to EVERY patch of the bag. Label preserved, within-bag structure
     (all patch-to-patch differences, hence the attention pattern) preserved exactly.

`strength` = offset size as a fraction of the observed between-ROI std (0.3 = a third of a typical
ROI-to-ROI nuisance shift). Composed variant with the champion CutMix: ``cutmix_then_bagshift``.
MPS-safe, deterministic given the global seed, no trainer edits.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._U: Optional[torch.Tensor] = None        # [r, D] orthonormal signal directions
        self._sigma: Optional[torch.Tensor] = None    # [1, D] between-bag (ROI) per-dim std
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        means = []
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            chunks[g].append(feats)
            means.append(feats.mean(dim=0))
        if len(chunks) < 2 or len(means) < 2:
            return
        M = torch.stack(means, dim=0)                                  # [B, D] bag means
        self._sigma = M.std(dim=0, keepdim=True).clamp(min=1e-6)       # [1, D]

        cents = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cents.append(allp.mean(dim=0))
        C = torch.stack(cents, dim=0)                                  # [G, D]
        Md = C - C.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(Md, full_matrices=False)
        rank = int((S > 1e-6 * S.max()).sum().item())
        self._U = Vh[:rank].contiguous()

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._U is None or self._sigma is None:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)                # [r, D]
        sig = self._sigma.to(features.device, features.dtype)          # [1, D]
        v = torch.randn_like(sig) * (sig * self.s)                     # [1, D] one offset for the bag
        v = v - (v @ U.t()) @ U                                        # keep only the nuisance part
        return features + v, float(label)
