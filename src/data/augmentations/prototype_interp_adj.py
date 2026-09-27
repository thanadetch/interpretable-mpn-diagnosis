"""prototype_interp_adj - ADJACENT-grade prototype interpolation (train-only). NEW: ordinal-
local variant of prototype_interp that protects the extreme grades.

prototype_interp picks ANY other grade as the interpolation target, so a G0 bag can be pulled
a t-fraction of the way toward the G3 centroid - which (at higher strength) collapses G0/G3
recall (observed: proto@0.4 crashed test G0 89->65). This variant restricts the target to an
ADJACENT grade only (|g_b - y| == 1): G0 moves only toward G1, G3 only toward G2, and middle
grades toward either neighbour. The shift therefore stays ordinal-local, enriching the G1/G2
boundary regions (where recall is weakest) without dragging extreme-grade bags across the
whole scale - keeping G0/G3 intact while still manufacturing intermediate-density bags.

`strength` = max interpolation fraction t toward the adjacent centroid. 0 disables. Requires
the train pool (per-grade centroids, computed once and cached). Fits scalar regression
(continuous target). Permutation/size-invariant, deterministic given the seed, MPS-safe.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.3):
        self.s = float(strength)
        self._proto: Optional[Dict[int, torch.Tensor]] = None
        self._tried = False

    def _ensure_proto(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            m = item[0].float().mean(dim=0)
            if g not in sums:
                sums[g] = torch.zeros_like(m)
                counts[g] = 0
            sums[g] += m
            counts[g] += 1
        if len(sums) < 2:
            return
        self._proto = {g: sums[g] / counts[g] for g in sums}

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None:
            return features, float(label)
        self._ensure_proto(pool)
        if self._proto is None:
            return features, float(label)
        y = int(round(float(label)))
        if y not in self._proto:
            return features, float(label)
        neighbours = [g for g in (y - 1, y + 1) if g in self._proto]
        if not neighbours:
            return features, float(label)
        gb = neighbours[int(torch.randint(len(neighbours), (1,)).item())]
        t = float(torch.rand(1).item()) * self.s
        direction = (self._proto[gb] - self._proto[y]).to(features.device, dtype=features.dtype)
        features = features + t * direction
        target = min(3.0, max(0.0, float(label) + t * (gb - y)))
        return features, target
