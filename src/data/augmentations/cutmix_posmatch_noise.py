"""cutmix_posmatch_noise - POSITION-MATCHED patch transplant + noise (train-only).

Third module on the spatial axis, and the one that isolates POSITION from CONTIGUITY. The champion
transplants a random donor patch into a random slot; ``cutmix_spatial_noise`` transplants a contiguous
block; this one keeps the champion's scattered, high-fraction replacement but requires every donor patch
to come from the SAME (row, col) grid position of a same-grade donor ROI.

Why position could matter even to a permutation-invariant aggregator: the patch grid is not
exchangeable in practice - ROIs are framed by the operator, so central positions tend to hold the tissue
of interest and border positions hold edge/background/artefact. Position-matched donors therefore keep
the "role" of each slot (centre stays centre, border stays border) while still randomising the tissue,
whereas the champion mixes border tissue into central slots and vice versa.

    replace a ``strength`` fraction of the bag; each chosen slot (r, c) receives the patch at (r, c) of a
    random same-grade donor ROI (falls back to a random donor patch when that donor lacks the position),
    then feature_noise @ 0.05

If both this and the contiguous-block variant lose to the champion's fully random transplant, the spatial
axis - the last structural blind spot of the registry - is closed, and patch position provably carries no
usable signal for this task. `strength` = replacement fraction. Uses ``rc`` (already in every .pt); no new
data, no trainer edits. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

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

    def __init__(self, strength: float = 0.8):
        self.s = float(strength)
        self._by_pos: Dict[int, Dict[Tuple[int, int], List[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        self._fp2rc: Dict[Tuple, torch.Tensor] = {}
        self._flat: Dict[int, torch.Tensor] = {}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        flat: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            flat[g].append(feats)
            path = _path_of(pool, i)
            rc = _load_rc(path) if path is not None else None
            if rc is None or rc.shape[0] != feats.shape[0]:
                continue
            self._fp2rc[_fp(feats)] = rc
            for j in range(rc.shape[0]):
                self._by_pos[g][(int(rc[j, 0]), int(rc[j, 1]))].append(feats[j])
        self._flat = {g: torch.cat([f for f in lst], dim=0) for g, lst in flat.items()}
        # stack per-position lists once
        for g in list(self._by_pos.keys()):
            self._by_pos[g] = {k: torch.stack(v, dim=0) for k, v in self._by_pos[g].items()}

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            g = int(round(float(label)))
            rc = self._fp2rc.get(_fp(features))
            posbank = self._by_pos.get(g)
            flat = self._flat.get(g)
            if rc is not None and posbank:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                out = features.clone()
                dev, dt = features.device, features.dtype
                for j in dst.tolist():
                    bank = posbank.get((int(rc[j, 0]), int(rc[j, 1])))
                    if bank is None or bank.shape[0] == 0:
                        if flat is None:
                            continue
                        bank = flat
                    out[j] = bank[int(torch.randint(bank.shape[0], (1,)).item())].to(dev, dt)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
