"""axis_quantile_warp - mean-fixed stretch/compress of patches ALONG the fibrosis axis
(train-only). NEW: centroid-locked axial warp (no mean move, no relabel).

Moves each patch along the fibrosis direction v by (k-1)*r_i, where r_i = (h_i . v) - bagmean_proj
is the patch's signed distance from the bag-mean along v, and k = 1 +/- U(0, strength) is a random
warp gain. Patches at the bag-mean don't move; the tails are stretched (k>1) or compressed (k<1).
Because mean_i[(k-1)*r_i] = 0, the bag-mean projection is UNCHANGED -> centroid locked along the
axis; target unchanged. Distinct from intrabag_spread (isotropic gaussian jitter, no axis, no fixed
point) and fibrosis_axis_shift (rigid translation of all patches + relabel). Lets the head see
varied within-bag density GRADIENTS at identical bag-mean and label.

`strength` = max warp magnitude |k-1|. 0 disables. Requires the train pool (axis v, cached once).
Permutation/size-invariant, deterministic given the seed, MPS-safe (matmul/outer only), no deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5):
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
        if self.s <= 0.0 or pool is None or features.shape[0] < 2:
            return features, float(label)
        self._ensure(pool)
        if self._v is None:
            return features, float(label)
        v = self._v.to(features.device, dtype=features.dtype)
        proj = features @ v
        r = proj - proj.mean()
        sign = 1.0 if float(torch.rand(1).item()) < 0.5 else -1.0
        k = 1.0 + float(torch.rand(1).item()) * self.s * sign
        new = features + ((k - 1.0) * r).unsqueeze(1) * v.unsqueeze(0)
        return new, float(label)
