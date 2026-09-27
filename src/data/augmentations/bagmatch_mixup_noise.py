"""bagmatch_mixup_noise - structure-preserving same-grade bag mixup + noise (train-only).

Prior bag-mixup variants either concatenated/subsampled two bags (naive) or interpolated across grades
(ordinal, collapsed). This does manifold mixup that RESPECTS bag structure: pick a random SAME-grade donor
bag, greedily match each original patch to its nearest donor patch (NN assignment), and move each patch a
fraction lambda toward its matched partner. Because matches are nearest-neighbours, the interpolation stays
on the local same-grade manifold (no off-manifold blends), and the bag keeps its size and per-patch identity.
lambda ~ U(0, strength) per bag. Label kept hard (both bags are the same grade). Then champion isotropic
noise. `strength` = max interpolation fraction. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bags: int = 4000):
        self.s = float(strength)
        self.max_bags = int(max_bags)
        self._bags: Optional[Dict[int, List[torch.Tensor]]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        bags: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            bags[g].append(item[0].float())
        for g in list(bags.keys()):
            if len(bags[g]) > self.max_bags:
                bags[g] = bags[g][: self.max_bags]
        self._bags = bags

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bags is None:
            return features, float(label)
        g = int(round(float(label)))
        lst = self._bags.get(g)
        if not lst or len(lst) < 2:
            return features, float(label)
        # pick a random same-grade donor bag (avoid degenerate self-match when possible)
        donor = lst[int(torch.randint(len(lst), (1,)).item())].to(features.device, features.dtype)
        if donor.shape[0] < 1:
            return features, float(label)
        # NN match each original patch to a donor patch
        d = torch.cdist(features, donor)                 # [N, M]
        nn = d.argmin(dim=1)                             # [N]
        matched = donor[nn]                              # [N, D]
        lam = float(torch.rand(1).item()) * self.s
        out = (1.0 - lam) * features + lam * matched
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
