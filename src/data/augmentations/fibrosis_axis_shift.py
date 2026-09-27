"""fibrosis_axis_shift - translate the whole bag along the LEARNED fibrosis direction,
with a grade-calibrated target shift (train-only). NEW axis: directional + label-aware.

All prior augmentations are direction-agnostic (drop/mix/resample/noise). This one is the
grading-principled augmentation: it estimates, once from the train pool, the fibrosis
direction ``v = mean(bag-mean | G2,G3) - mean(bag-mean | G0,G1)`` (the coverage/density
axis the thesis shows is monotone in grade), and the slope ``a = d(grade)/d(projection
onto v)``. To augment a bag, it draws a grade-space shift ``dg ~ Normal(0, strength)``,
translates EVERY patch by ``(dg / a) * v_unit`` (so the bag's projection moves by exactly
``dg``), and moves the regression target to ``clip(label + dg, 0, 3)``. This manufactures
plausible higher/lower-density versions of a bag along the exact axis grade tracks - a
density-interpolation augmentation grounded in the thesis interpretability finding.

`strength` = std of the grade-space shift dg (in grade units). 0 disables. Requires the
train pool (to estimate v and a, computed once and cached). Fits scalar regression
(continuous target). Permutation/size-invariant (adds the same vector to all patches),
deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.3):
        self.s = float(strength)
        self._v: Optional[torch.Tensor] = None  # unit fibrosis direction [D] (cpu)
        self._a: Optional[float] = None          # slope d(grade)/d(projection)
        self._tried = False

    def _ensure_axis(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        means, grades = [], []
        for i in range(len(pool)):
            item = pool[i]
            f = item[0]
            means.append(f.float().mean(dim=0))  # bag-mean [D]
            grades.append(float(item[1]))
        if len(means) < 4:
            return
        M = torch.stack(means)                    # [B, D]
        g = torch.tensor(grades)                  # [B]
        hi_mask = g >= 2.0
        lo_mask = g <= 1.0
        if int(hi_mask.sum()) == 0 or int(lo_mask.sum()) == 0:
            return
        v = M[hi_mask].mean(dim=0) - M[lo_mask].mean(dim=0)
        v = v / (v.norm() + 1e-8)
        proj = M @ v                              # [B]
        pc = proj - proj.mean()
        gc = g - g.mean()
        denom = (pc * pc).sum()
        if float(denom) < 1e-8:
            return
        a = float((pc * gc).sum() / denom)
        if abs(a) < 1e-6:
            return
        self._v = v
        self._a = a

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None:
            return features, float(label)
        self._ensure_axis(pool)
        if self._v is None or self._a is None:
            return features, float(label)
        dg = float(torch.randn(1).item()) * self.s          # grade-space shift
        shift = dg / self._a                                # projection-space shift
        v = self._v.to(features.device, dtype=features.dtype)
        features = features + shift * v                     # translate whole bag along v
        target = min(3.0, max(0.0, float(label) + dg))
        return features, target
