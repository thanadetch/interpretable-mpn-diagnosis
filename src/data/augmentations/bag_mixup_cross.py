"""bag_mixup_cross - canonical cross-grade MixUp at the bag level (train-only, regression).

Standard MixUp (Zhang et al. 2018) adapted to MIL: mix the current bag with a random OTHER bag and
interpolate the target. A fraction lam of the resulting bag's patches come from the current bag and
(1-lam) from the donor bag, and the regression target becomes lam*g_self + (1-lam)*g_donor. Unlike
cutmix (same-grade, hard label) this crosses grades with a soft interpolated target = the textbook
MixUp regulariser, testing whether label-interpolation helps the ordinal head.

    donor bag ~ ANY grade ;  lam ~ Beta(ALPHA, ALPHA)
    keep round(lam*N) self patches + round((1-lam)*N) donor patches ;  y = lam*g_self + (1-lam)*g_donor
    label = interpolated (requires_regression).

`strength` = alpha (Beta concentration). Requires the train pool (whole bags). MPS-safe, deterministic.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.4)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.4, max_bags: int = 800):
        self.a = float(strength)
        self.max_bags = int(max_bags)
        self._bags: Optional[List[torch.Tensor]] = None
        self._labels: Optional[List[float]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        bags, labels = [], []
        idxs = list(range(len(pool)))
        if len(idxs) > self.max_bags:
            idxs = torch.randperm(len(pool))[: self.max_bags].tolist()
        for i in idxs:
            item = pool[i]
            bags.append(item[0].float())
            labels.append(float(item[1]))
        self._bags, self._labels = bags, labels

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.a <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if not self._bags:
            return features, float(label)
        j = int(torch.randint(len(self._bags), (1,)).item())
        donor = self._bags[j].to(features.device, features.dtype)
        g_donor = self._labels[j]
        lam = float(torch.distributions.Beta(self.a, self.a).sample().item())
        n = features.shape[0]
        ks = min(max(1, int(round(lam * n))), n - 1)          # self patches
        kd = n - ks                                           # donor patches
        si = torch.randperm(n)[:ks]
        di = torch.randint(donor.shape[0], (kd,))
        out = torch.cat([features[si], donor[di]], dim=0)
        y = lam * float(label) + (1.0 - lam) * g_donor
        return out, float(y)
