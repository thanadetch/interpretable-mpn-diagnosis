"""cutmix_spatial_noise - TRUE 2-D CutMix on the patch grid (contiguous block) + noise (train-only).

Blind spot of the whole 144-module registry: **nothing uses the patch coordinates**, even though every
feature file stores ``rc`` = the (row, col) position of each patch in its ROI grid (5x8, 7x6, 7x16,
4x10 ... depending on the ROI). Every existing CutMix variant replaces a random, spatially SCATTERED
subset of patches - which is not what CutMix does in vision: the original replaces a CONTIGUOUS
rectangular region, and that is what preserves local structure inside the transplanted region.

That distinction matters here specifically: reticulin grading is about the CONTINUITY of the fibre network
across the field (the WHO criterion separating G1 from G0/G2), and the global-view round independently
showed that field-level continuity information lifts G1 recall. A scattered transplant destroys the local
neighbourhood relations of both the recipient and the donor; a block transplant keeps a real, spatially
coherent piece of donor tissue intact and only breaks continuity at the block boundary.

    1. sample a rectangle in the recipient's (row, col) grid with area ~ ``strength`` (CutMix-style:
       h = R*sqrt(s), w = C*sqrt(s), random top-left)
    2. pick ONE random same-grade donor ROI and copy its patches at the SAME grid positions into that
       rectangle (positions the donor lacks fall back to a random donor patch)
    3. feature_noise @ 0.05

`strength` = area fraction of the block. Falls back to uniform random replacement if coordinates are
unavailable. No new data (rc is already in every .pt), no trainer edits. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

SIGMA = 0.05


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

    def __init__(self, strength: float = 0.5):
        self.s = float(strength)
        # per grade: list of (feats [N,D], {(r,c): row index})
        self._rois: Dict[int, List[Tuple[torch.Tensor, Dict[Tuple[int, int], int]]]] = defaultdict(list)
        self._fp2rc: Dict[Tuple, torch.Tensor] = {}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            path = _path_of(pool, i)
            rc = _load_rc(path) if path is not None else None
            if rc is None or rc.shape[0] != feats.shape[0]:
                continue
            self._fp2rc[_fp(feats)] = rc
            pos = {(int(rc[j, 0]), int(rc[j, 1])): j for j in range(rc.shape[0])}
            self._rois[g].append((feats, pos))

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            g = int(round(float(label)))
            rc = self._fp2rc.get(_fp(features))
            donors = self._rois.get(g)
            if rc is not None and donors:
                dfeat, dpos = donors[int(torch.randint(len(donors), (1,)).item())]
                if dfeat.shape[1] == features.shape[1]:
                    R = int(rc[:, 0].max().item()) + 1
                    C = int(rc[:, 1].max().item()) + 1
                    frac = self.s ** 0.5
                    bh = max(1, min(R, int(round(R * frac))))
                    bw = max(1, min(C, int(round(C * frac))))
                    r0 = int(torch.randint(max(1, R - bh + 1), (1,)).item())
                    c0 = int(torch.randint(max(1, C - bw + 1), (1,)).item())
                    inside = ((rc[:, 0] >= r0) & (rc[:, 0] < r0 + bh) &
                              (rc[:, 1] >= c0) & (rc[:, 1] < c0 + bw)).nonzero(as_tuple=True)[0]
                    if inside.numel() > 0:
                        out = features.clone()
                        dev, dt = features.device, features.dtype
                        for j in inside.tolist():
                            key = (int(rc[j, 0]), int(rc[j, 1]))
                            k = dpos.get(key)
                            if k is None:
                                k = int(torch.randint(dfeat.shape[0], (1,)).item())
                            out[j] = dfeat[k].to(dev, dt)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
