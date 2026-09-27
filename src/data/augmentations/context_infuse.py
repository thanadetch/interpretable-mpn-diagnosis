"""context_infuse - blend every patch toward its OWN ROI's whole-view embedding (train-only).

Ablation partner of ``cutmix_globalview_noise``: that module injects the global view as DONOR TOKENS from
other ROIs (adding tokens of a different magnification to the bag); this one adds the same information as
CONTEXT to the patches the bag already has, without introducing any foreign token:

    h_i  <-  (1 - strength) * h_i  +  strength * g_own

where g_own is the whole-ROI (no_patch) embedding of the SAME ROI, from the same encoder and feature space.
Each patch keeps its own identity (the bag's within-bag structure is preserved up to a common shift toward
a shared vector) but is told what field it came from.

The two together separate "does the global view help at all?" from "does it help as extra tokens?".
Note the risk this one deliberately probes: pulling all patches toward one shared vector shrinks within-bag
variance, which killed prototype_shrink - the difference here is that the anchor is the ROI's OWN real
whole-field embedding, not a grade centroid, so it does not erase between-ROI discriminative variation.

`strength` = blend weight (0.2 default). Requires ``data/features_<backbone>_reti_no_patch`` (already on
disk for all 3 backbones); the recipient ROI is identified by a feature-value fingerprint. No new data,
no trainer edits. MPS-safe, deterministic.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


def _path_of(pool, i: int) -> Optional[Path]:
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        return ds.samples[idxs[i]][0] if (ds is not None and idxs is not None) else pool.samples[i][0]
    except Exception:
        return None


def _no_patch_path(p: Path) -> Optional[Path]:
    parts = list(p.parts)
    for j, part in enumerate(parts):
        if part.startswith("features_") and not part.endswith("_no_patch"):
            parts[j] = part + "_no_patch"
            return Path(*parts)
    return None


def _fp(features: torch.Tensor) -> Tuple:
    n, d = features.shape
    return (int(n), int(d), float(features[0, 0]), float(features[0, -1]),
            float(features[-1, 0]), float(features[-1, -1]))


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2):
        self.s = float(strength)
        self._fp2g: Dict[Tuple, torch.Tensor] = {}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        for i in range(len(pool)):
            feats = pool[i][0].float()
            path = _path_of(pool, i)
            gp = _no_patch_path(path) if path is not None else None
            if gp is None or not gp.exists():
                continue
            try:
                data = torch.load(gp, map_location="cpu", weights_only=False)
                gv = data["feats"] if isinstance(data, dict) else data
                gv = gv.float().reshape(1, -1)
            except Exception:
                continue
            if gv.shape[1] == feats.shape[1]:
                self._fp2g[_fp(feats)] = gv

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        gv = self._fp2g.get(_fp(features))
        if gv is None:
            return features, float(label)
        g = gv.to(features.device, features.dtype)
        return (1.0 - self.s) * features + self.s * g, float(label)
