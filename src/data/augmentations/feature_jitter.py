"""feature_jitter - per-bag channel-wise affine domain randomization (train-only).

A genuinely different lever from CutMix (swap patches) and per-patch noise (perturb patches independently):
here the WHOLE bag is passed through one random per-channel affine transform that is SHARED across all
patches in the bag. This mimics a bag-level "style" / scanner / stain shift in feature space — the kind of
nuisance variation that separates one patient (val) from another (test) on the val<->test −0.95 axis.
Randomising it per bag pushes the model to be invariant to that shift.

    per bag:  scale_d ~ N(1, strength^2),  shift_d ~ N(0, (strength * global_std_d)^2)   (per channel)
    h'_i = h_i * scale + shift            (same scale/shift for every patch i in the bag)   label kept.

Unlike MixStyle (which swaps instance stats between bags), this RANDOMISES the per-channel gain/bias, a
domain-randomisation regulariser. No CutMix -> titan-safe. `strength` = jitter magnitude. Requires the
train pool only for the global per-dim std. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.1)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.1, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._std: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        rows = []
        for i in range(len(pool)):
            rows.append(pool[i][0].float())
        allp = torch.cat(rows, dim=0)
        if allp.shape[0] > self.max_bank:
            allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
        self._std = allp.std(dim=0, keepdim=True).clamp(min=1e-6)   # [1, D] global per-dim std

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._std is None:
            return features, float(label)
        d = features.shape[1]
        sd = self._std.to(features.device, features.dtype)
        scale = 1.0 + self.s * torch.randn(1, d, device=features.device, dtype=features.dtype)
        shift = self.s * sd * torch.randn(1, d, device=features.device, dtype=features.dtype)
        return features * scale + shift, float(label)
