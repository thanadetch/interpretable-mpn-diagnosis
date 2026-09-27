"""axis_g1_gentle - fibrosis-axis shift + a TUNABLE-magnitude G1 nudge for every G0/G2 bag.
NEW: the minimal-nudge probe of the val<->G1 frontier.

axis_g1_hybrid showed the G1 nudge is exactly what trades val for G1: pure axis@0.2 has val 0.8113
/ G1 71; adding the nudge (G1_MAX=0.3) lifts G1 to ~85 but drops val to ~0.785. This variant keeps
the proven axis@0.2 base, ALWAYS nudges G0/G2 bags toward the G1 centroid, but exposes the nudge
MAGNITUDE as ``strength`` (= max fraction t) so we can find the gentlest nudge that pulls G1 just
over 78 while keeping val > 0.7968 - i.e. probe whether the val<->G1 frontier has ANY threadable
point.

`strength` = max G1-ward nudge fraction t (drawn U(0, strength)); the axis component is fixed at
s_axis=0.2. 0 disables the nudge (= pure fibrosis_axis_shift@0.2). Requires the train pool (axis +
centroids, cached once). Fits scalar regression. Permutation/size-invariant, deterministic given
the seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.15)
_S_AXIS = 0.2


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.15):
        self.g1max = float(strength)
        self._v: Optional[torch.Tensor] = None
        self._a: Optional[float] = None
        self._proto: Optional[Dict[int, torch.Tensor]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        means, grades = [], []
        sums: Dict[int, torch.Tensor] = {}
        counts: Dict[int, int] = {}
        for i in range(len(pool)):
            item = pool[i]
            m = item[0].float().mean(dim=0)
            g = int(round(float(item[1])))
            means.append(m)
            grades.append(float(item[1]))
            if g not in sums:
                sums[g] = torch.zeros_like(m)
                counts[g] = 0
            sums[g] += m
            counts[g] += 1
        if len(means) < 4 or len(sums) < 2:
            return
        M = torch.stack(means)
        gt = torch.tensor(grades)
        hi = gt >= 2.0
        lo = gt <= 1.0
        if int(hi.sum()) and int(lo.sum()):
            v = M[hi].mean(dim=0) - M[lo].mean(dim=0)
            v = v / (v.norm() + 1e-8)
            proj = M @ v
            pc = proj - proj.mean()
            denom = (pc * pc).sum()
            if float(denom) >= 1e-8:
                a = float((pc * (gt - gt.mean())).sum() / denom)
                if abs(a) >= 1e-6:
                    self._v = v
                    self._a = a
        self._proto = {g: sums[g] / counts[g] for g in sums}

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None:
            return features, float(label)
        self._ensure(pool)
        target = float(label)
        if self._v is not None and self._a is not None:
            dg = float(torch.randn(1).item()) * _S_AXIS
            features = features + (dg / self._a) * self._v.to(features.device, dtype=features.dtype)
            target += dg
        if self.g1max > 0.0 and self._proto is not None and 1 in self._proto:
            y = int(round(float(label)))
            if y in (0, 2) and y in self._proto:
                t = float(torch.rand(1).item()) * self.g1max
                direction = (self._proto[1] - self._proto[y]).to(features.device, dtype=features.dtype)
                features = features + t * direction
                target += t * (1 - y)
        return features, min(3.0, max(0.0, target))
