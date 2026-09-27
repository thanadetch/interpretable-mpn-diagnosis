"""centroid_locked_intrabag_resample - resample patches by fibrosis-axis projection, then
RE-CENTER to the original bag-mean and KEEP the label (train-only). NEW mechanism class:
mean-AND-label-preserving within-bag redistribution.

Every prior augmentation moved the bag-mean (axis/proto/smote/spread/coverage) or relabeled
(boundary/hybrid/g1_synth/proto). This one changes ONLY the within-bag patch multiplicities:
resample N patches with weights softmax(gamma*z) where z is the standardized projection onto the
fibrosis axis v (gamma ~ N(0, strength), random sign), then add a constant offset so the resampled
bag's mean exactly equals the original bag-mean. Target is unchanged. The attention head therefore
sees the SAME centroid but a varying sub-population emphasis, forcing it to localize the
high-fibrosis sub-population that distinguishes the grade rather than reading the mean -
the one mechanism whose argument for lifting G1 does NOT shift prediction mass toward the middle
(extreme bags keep their extreme mean, so the val extremes are protected).

`strength` = std of gamma (resampling sharpness). 0 disables. Requires the train pool (axis v,
cached once). Fits scalar regression. Permutation/size-invariant, deterministic given the seed,
MPS-safe (multinomial on CPU), no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.6)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.6):
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
        self._ensure(pool)
        if self._v is None:
            return features, float(label)
        v = self._v.to(features.device, dtype=features.dtype)
        proj = features @ v
        z = (proj - proj.mean()) / (proj.std() + 1e-6)
        gamma = float(torch.randn(1).item()) * self.s
        w = torch.softmax(gamma * z, dim=0)
        idx = torch.multinomial(w.detach().cpu(), features.shape[0], replacement=True).to(features.device)
        new = features[idx]
        new = new + (features.mean(dim=0, keepdim=True) - new.mean(dim=0, keepdim=True))  # mean-lock
        return new, float(label)
