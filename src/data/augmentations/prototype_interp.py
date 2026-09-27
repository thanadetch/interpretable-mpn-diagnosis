"""prototype_interp - translate a bag between GRADE PROTOTYPES (train-only). NEW: prototype-
guided, multi-directional, label-aware.

Generalises fibrosis_axis_shift from a single global axis to per-grade centroids. Once, from
the train pool, it estimates the prototype (pooled patch-mean) of each grade present:
``p_g = mean(bag-mean | grade == g)``. To augment a bag of grade ``y``, it picks a random
other grade ``g_b``, draws ``t ~ Uniform(0, strength)``, and translates EVERY patch by
``t * (p_{g_b} - p_y)`` - moving the bag a t-fraction of the way from its own grade centroid
toward another grade's centroid - and sets the target to ``clip(y + t*(g_b - y), 0, 3)``.
Unlike the single fibrosis axis, this follows the actual centroid-difference directions
between specific grade pairs (multi-directional density interpolation).

`strength` = max interpolation fraction t. 0 disables. Requires the train pool (prototypes
computed once and cached). Fits scalar regression (continuous target). Permutation/size-
invariant (same shift to all patches), deterministic given the global seed, MPS-safe, no deps.
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
        self._proto: Optional[Dict[int, torch.Tensor]] = None  # grade -> centroid [D] (cpu)
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
        others = [g for g in self._proto if g != y]
        if not others:
            return features, float(label)
        gb = others[int(torch.randint(len(others), (1,)).item())]
        t = float(torch.rand(1).item()) * self.s
        direction = (self._proto[gb] - self._proto[y]).to(features.device, dtype=features.dtype)
        features = features + t * direction
        target = min(3.0, max(0.0, float(label) + t * (gb - y)))
        return features, target
