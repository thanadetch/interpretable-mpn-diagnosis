"""feature_smote - SMOTE-style same-grade nearest-neighbour interpolation (train-only).
NEW: structured WITHIN-grade interpolation toward the NEAREST same-grade bag (not random, not
cross-grade).

Classic SMOTE synthesises minority examples by interpolating a sample toward a same-class nearest
neighbour. Here, for a bag of grade y, we find the NEAREST same-grade bag (by bag-mean L2 distance,
from cached per-grade means) and translate every patch of the current bag a fraction lam toward
that neighbour's centroid: h_i += lam * (nn_mean - own_mean). The grade is unchanged (same-class
interpolation, no label noise). This manufactures new on-manifold variants of each grade -
including the bag-scarce G1 - using the LOCAL same-grade structure, distinct from prototype_interp
(toward another-grade centroid), mixup_within_grade (random same-grade patch pool), and g1_synth
(cross-grade G0+G2).

`strength` = max interpolation fraction lam (drawn U(0, strength)). 0 disables. Target unchanged.
Requires the train pool (per-grade bag-means, cached once). Permutation/size-invariant,
deterministic given the seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5):
        self.s = float(strength)
        self._means_by_grade: Optional[Dict[int, torch.Tensor]] = None  # grade -> [K_g, D]
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        buckets: Dict[int, List[torch.Tensor]] = {}
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            buckets.setdefault(g, []).append(item[0].float().mean(dim=0))
        self._means_by_grade = {
            g: torch.stack(v) for g, v in buckets.items() if len(v) >= 2
        }

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None:
            return features, float(label)
        self._ensure(pool)
        y = int(round(float(label)))
        if not self._means_by_grade or y not in self._means_by_grade:
            return features, float(label)
        bank = self._means_by_grade[y].to(features.device, dtype=features.dtype)  # [K, D]
        own = features.mean(dim=0, keepdim=True)                                  # [1, D]
        d = torch.norm(bank - own, dim=1)                                         # [K]
        # nearest OTHER bag (smallest distance > 0); if all zero, fall back to argmin
        order = torch.argsort(d)
        nn = order[1] if (d.numel() > 1 and float(d[order[0]]) < 1e-6) else order[0]
        nn_mean = bank[nn].unsqueeze(0)                                           # [1, D]
        lam = float(torch.rand(1).item()) * self.s
        features = features + lam * (nn_mean - own)
        return features, float(label)
