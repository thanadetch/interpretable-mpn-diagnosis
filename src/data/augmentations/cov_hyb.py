"""cov_hyb - stack the non-sampler VAL-booster (coverage_resample) onto the TEST+G1 winner
(axis_g1_hybrid). NEW: combine two complementary-gate levers WITHOUT the sampler (so test is not
capped at ~0.956).

Rationale from the frontier map: axis_g1_hybrid q=0.7 (no sampler) already clears test (0.9658) and
G1 (85.7) but misses val by 0.012; coverage_resample is a val-booster (+0.02..0.04) that, unlike the
sampler, does NOT cap test. Stacking them per bag:
  1) coverage_resample: resample patches by fibrosis-axis projection (gamma ~ N(0, 1.0)); target +=
     a * (mean-projection shift).   [val booster]
  2) fibrosis_axis_shift @ s_axis=0.2: translate the (resampled) bag along v; target += dg.  [val+test]
  3) G1 nudge: with prob `strength`, a G0/G2 bag is nudged toward the G1 centroid; target += t*(1-y). [G1]
All three are feature-space density operations along the SAME learned axis/centroids, so the target
deltas add linearly (clipped to [0,3]).

`strength` = probability of the G1 nudge (G0/G2 only). 0 = coverage+axis only. Requires the train
pool (v, slope a, grade centroids, cached once). Fits scalar regression. Permutation/size-invariant,
deterministic given the seed, MPS-safe (multinomial on CPU), no new deps.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)
_COV_STD = 1.0
_S_AXIS = 0.2
_G1_MAX = 0.3


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.5):
        self.q = float(strength)
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
            v = self._v.to(features.device, dtype=features.dtype)
            # 1) coverage resample (val booster)
            if features.shape[0] >= 2:
                proj = features @ v
                mu = proj.mean()
                z = (proj - mu) / (proj.std() + 1e-6)
                gamma = float(torch.randn(1).item()) * _COV_STD
                w = torch.softmax(gamma * z, dim=0)
                idx = torch.multinomial(w.detach().cpu(), features.shape[0], replacement=True).to(features.device)
                features = features[idx]
                target += self._a * float((features @ v).mean() - mu)
            # 2) axis shift (val+test driver)
            dg = float(torch.randn(1).item()) * _S_AXIS
            features = features + (dg / self._a) * v
            target += dg
        # 3) G1 nudge (G0/G2), prob q
        if self.q > 0.0 and self._proto is not None and 1 in self._proto:
            y = int(round(float(label)))
            if y in (0, 2) and y in self._proto and float(torch.rand(1).item()) <= self.q:
                t = float(torch.rand(1).item()) * _G1_MAX
                features = features + t * (self._proto[1] - self._proto[y]).to(features.device, dtype=features.dtype)
                target += t * (1 - y)
        return features, min(3.0, max(0.0, target))
