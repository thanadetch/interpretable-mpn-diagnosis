"""cutout_spatial - drop a CONTIGUOUS region of the patch grid (Cutout analogue, train-only).

Second module on the previously unused SPATIAL axis (see ``cutmix_spatial_noise``). Where patch_dropout
removes randomly SCATTERED patches (destructive: it thins the whole field uniformly, so the bag keeps a
faithful but noisier sample of every region), this removes a contiguous rectangular BLOCK - the Cutout /
random-erasing analogue, and a realistic one: it simulates an ROI where part of the field is unusable
(fold, tear, out-of-focus corner, marker ink), which is exactly what the rejected-patch mask produces in
real cases.

    sample a rectangle of area ~ ``strength`` in the recipient's (row, col) grid and DROP those patches
    (bag shrinks; no donors, no noise - a pure structural ablation)

It also probes the aggregators' spatial-coverage assumption directly: if performance holds, the grade is
readable from any sufficiently large contiguous sub-field (good news for robustness, and evidence the
model is not relying on a specific location); if it collapses, the grade decision depends on FULL field
coverage - which would be an argument the patch-level attention story needs to address.

`strength` = area fraction removed (0.3 default). Requires ``rc`` (already in every .pt); no-ops when
coordinates are unavailable. No trainer edits. MPS-safe, deterministic given the global seed.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.3)

MIN_KEEP = 4


def _path_of(pool, i: int) -> Optional[Path]:
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        return ds.samples[idxs[i]][0] if (ds is not None and idxs is not None) else pool.samples[i][0]
    except Exception:
        return None


def _load_rc(p: Path) -> Optional[torch.Tensor]:
    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        rc = data.get("rc") if isinstance(data, dict) else None
        return torch.as_tensor(rc).long() if rc is not None else None
    except Exception:
        return None


def _fp(features: torch.Tensor) -> Tuple:
    n, d = features.shape
    return (int(n), int(d), float(features[0, 0]), float(features[0, -1]),
            float(features[-1, 0]), float(features[-1, -1]))


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.3):
        self.s = float(strength)
        self._fp2rc: Dict[Tuple, torch.Tensor] = {}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        for i in range(len(pool)):
            feats = pool[i][0].float()
            path = _path_of(pool, i)
            rc = _load_rc(path) if path is not None else None
            if rc is not None and rc.shape[0] == feats.shape[0]:
                self._fp2rc[_fp(feats)] = rc

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] <= MIN_KEEP:
            return features, float(label)
        self._ensure(pool)
        rc = self._fp2rc.get(_fp(features))
        if rc is None:
            return features, float(label)
        R = int(rc[:, 0].max().item()) + 1
        C = int(rc[:, 1].max().item()) + 1
        frac = self.s ** 0.5
        bh = max(1, min(R, int(round(R * frac))))
        bw = max(1, min(C, int(round(C * frac))))
        r0 = int(torch.randint(max(1, R - bh + 1), (1,)).item())
        c0 = int(torch.randint(max(1, C - bw + 1), (1,)).item())
        inside = ((rc[:, 0] >= r0) & (rc[:, 0] < r0 + bh) &
                  (rc[:, 1] >= c0) & (rc[:, 1] < c0 + bw))
        keep = (~inside).nonzero(as_tuple=True)[0]
        if keep.numel() < MIN_KEEP:
            return features, float(label)
        return features[keep.to(features.device)], float(label)
