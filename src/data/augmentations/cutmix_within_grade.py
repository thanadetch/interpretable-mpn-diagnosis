"""cutmix_within_grade - discrete cross-bag SAME-GRADE patch transplant (train-only). NEW
mechanism: discrete, on-manifold patch swap (no blending, no centroid math, label-locked).

For a bag of grade ``y``, replace a ``strength`` fraction of its patches with REAL patches drawn
(uniformly) from the pooled patch-bank of OTHER same-grade bags. The transplanted patches are
genuine, unmodified foundation-model features (on-manifold), so no synthetic/off-manifold artefact
is introduced; the grade is unchanged so the target stays ``y``. This recombines the within-grade
patch population into new plausible bag compositions, increasing per-grade sample diversity.

Distinct from every prior augmentation: mixup_* / multimix / patch_mixup_self BLEND patches
(convex combination); feature_smote interpolates toward a same-grade neighbour; bag_bootstrap
resamples WITHIN the same bag; prototype_interp / fibrosis_axis_shift TRANSLATE along centroid
directions. Here patches are swapped DISCRETELY across same-grade bags with no arithmetic on the
features. Mean stays inside the grade cloud (donor is same-grade), so it does not systematically
push toward the extremes.

`strength` = fraction of a bag's patches replaced (0 disables). Requires the train pool (per-grade
patch banks cached once, capped at ``max_bank`` patches/grade for memory). Permutation/size-
invariant, MPS-safe (indexing only), no new deps.
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
        self._bank: Optional[Dict[int, torch.Tensor]] = None  # grade -> [M, D] (cpu)
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
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)  # [M, D]
            if allp.shape[0] > self.max_bank:
                sel = torch.randperm(allp.shape[0])[: self.max_bank]
                allp = allp[sel]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = max(1, int(round(self.s * n)))
        k = min(k, n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        new = features.clone()
        new[dst] = bank[src].to(features.device, features.dtype)
        return new, float(label)
