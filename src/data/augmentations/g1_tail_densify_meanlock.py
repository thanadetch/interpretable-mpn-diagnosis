"""g1_tail_densify_meanlock - G1-only, centroid-locked + label-locked within-bag tail enrichment
(train-only). NEW: the explicit attempt to lift G1 while defeating the val wall by never moving
the G1 bag's mean or label and never touching G0/G2/G3.

Fires ONLY on G1 bags. Resamples patches with weights softmax(gamma*z) (gamma = |N(0,strength)|,
over-representing the moderate-density tail that characterizes mild fibrosis), then RE-CENTERS the
resampled bag onto the original bag-mean and keeps target = 1.0. So the G1 bag's centroid and label
are both unchanged; only the internal G1 tail is enriched. Distinct from g1_synth_mixup (synthesises
G1 from G0+G2 cross-grade + relabels) and boundary_contrast (fractional 0.5/1.5 targets): here the
bag STAYS G1, target STAYS 1.0, mean STAYS fixed. This cleanly tests whether G1 recall can be lifted
by internal redistribution alone (mass-shift removed as a confound) - if it still trips val, the val
wall is cohort-geometry, not centroid arithmetic.

`strength` = std of gamma (tail sharpness). 0 disables. Requires the train pool (axis v, cached
once). Fits scalar regression. Permutation/size-invariant, deterministic given the seed, MPS-safe
(multinomial on CPU), no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.4)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.4):
        self.s = float(strength)
        self._v: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        means, grades = [], []
        for i in range(len(pool)):
            item = pool[i]
            means.append(item[0].float().mean(dim=0))
            grades.append(float(item[1]))
        if len(means) < 4:
            return
        M = torch.stack(means)
        gt = torch.tensor(grades)
        hi = gt >= 2.0
        lo = gt <= 1.0
        if int(hi.sum()) == 0 or int(lo.sum()) == 0:
            return
        v = M[hi].mean(dim=0) - M[lo].mean(dim=0)
        self._v = v / (v.norm() + 1e-8)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 3:
            return features, float(label)
        if int(round(float(label))) != 1:
            return features, float(label)
        self._ensure(pool)
        if self._v is None:
            return features, float(label)
        v = self._v.to(features.device, dtype=features.dtype)
        proj = features @ v
        z = (proj - proj.mean()) / (proj.std() + 1e-6)
        gamma = abs(float(torch.randn(1).item())) * self.s
        w = torch.softmax(gamma * z, dim=0)
        idx = torch.multinomial(w.detach().cpu(), features.shape[0], replacement=True).to(features.device)
        new = features[idx]
        new = new + (features.mean(dim=0, keepdim=True) - new.mean(dim=0, keepdim=True))  # mean-lock
        return new, 1.0
