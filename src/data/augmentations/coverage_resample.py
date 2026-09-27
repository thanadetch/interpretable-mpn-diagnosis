"""coverage_resample - resample patches by their fibrosis-axis projection to perturb COVERAGE
(train-only). NEW mechanism class: distribution-resampling along the grade axis, not translation.

The thesis signal is coverage/density along the learned fibrosis direction
v = mean(G2,G3) - mean(G0,G1). Every prior augmentation translates/mixes/drops/scales features;
this instead RESAMPLES the bag's own patches (with replacement, size preserved) using weights
softmax(gamma * z_i), where z_i is the standardized projection of patch i onto v_unit and
gamma ~ N(0, strength). gamma>0 over-represents high-fibrosis patches (simulates a denser ROI),
gamma<0 under-represents them (sparser). The realized mean-projection change dp is mapped to the
target via the calibrated slope a = d(grade)/d(projection): target = clip(label + a*dp, 0, 3).
So it manufactures plausible higher/lower-coverage versions of a real bag along the exact axis the
grade tracks - a coverage-space augmentation, complementary to fibrosis_axis_shift (which rigidly
translates ALL patches; this changes the MIX of patches).

`strength` = std of gamma (the resampling sharpness). 0 disables. Requires the train pool (v + a,
cached once). Fits scalar regression. Permutation/size-invariant, deterministic given the seed,
MPS-safe (multinomial sampling done on CPU then moved), no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=1.0)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 1.0):
        self.s = float(strength)
        self._v: Optional[torch.Tensor] = None
        self._a: Optional[float] = None
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
        v = v / (v.norm() + 1e-8)
        proj = M @ v
        pc = proj - proj.mean()
        denom = (pc * pc).sum()
        if float(denom) < 1e-8:
            return
        a = float((pc * (gt - gt.mean())).sum() / denom)
        if abs(a) < 1e-6:
            return
        self._v = v
        self._a = a

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 2:
            return features, float(label)
        self._ensure(pool)
        if self._v is None or self._a is None:
            return features, float(label)
        v = self._v.to(features.device, dtype=features.dtype)
        proj = features @ v                                    # [N]
        mu = proj.mean()
        z = (proj - mu) / (proj.std() + 1e-6)
        gamma = float(torch.randn(1).item()) * self.s
        w = torch.softmax(gamma * z, dim=0)
        N = features.shape[0]
        idx = torch.multinomial(w.detach().cpu(), N, replacement=True).to(features.device)
        new = features[idx]
        dp = float((new @ v).mean() - mu)                      # projection-space coverage shift
        target = min(3.0, max(0.0, float(label) + self._a * dp))
        return new, target
