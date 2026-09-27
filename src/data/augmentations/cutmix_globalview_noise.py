"""cutmix_globalview_noise - CutMix donors from the WHOLE-ROI (low-magnification) view + noise (train-only).

Every one of the ~130 registry modules re-arranges the SAME information: the patch-level features of the
train ROIs. That is why they all land on the same val<->test frontier - the patch view is the ceiling.
This module is the first to inject information the patch view structurally cannot contain.

`data/features_<backbone>_reti_no_patch/` holds, for all 1330 ROIs and all three backbones, the embedding
of the WHOLE ROI resized (no patching) - produced by the SAME encoder, so it lives in the same feature
space and has the same dimension (uni2 1536 / virchow2 1280 / titan 768) and a comparable norm
(15.8 vs 13.98 for uni2), but it is a genuinely different magnification: it encodes the global fibre
ARCHITECTURE of the field (density and continuity across the whole ROI, cos with the patch mean ~0.65 =
related but far from redundant), which is exactly the WHO criterion that separates G2/G3 (diffuse dense
network, coarse bundles) and which no single patch can express.

    replace a ``strength`` fraction of the bag's patches with whole-ROI embeddings of OTHER TRAIN ROIs of
    the SAME grade, then feature_noise @ 0.05

So the bag becomes a mixture of local-texture tokens and global-architecture tokens, all same-grade and
all real encoder outputs. Train-only (test bags stay pure patch view), exactly like every CutMix variant
here - the augmented bag is synthetic either way. Donors come ONLY from the train pool (the no_patch file
is resolved from each train ROI's own path), so there is no val/test leakage. No new data, no extraction,
no trainer edits. `strength` = fraction of patches replaced by the global view. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)

SIGMA = 0.05


def _path_of(pool, i: int) -> Optional[Path]:
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        return ds.samples[idxs[i]][0] if (ds is not None and idxs is not None) else pool.samples[i][0]
    except Exception:
        return None


def _no_patch_path(p: Path) -> Optional[Path]:
    """data/features_X_reti/ET/ET6 G1/reti1.pt -> data/features_X_reti_no_patch/ET/ET6 G1/reti1.pt"""
    parts = list(p.parts)
    for j, part in enumerate(parts):
        if part.startswith("features_") and not part.endswith("_no_patch"):
            parts[j] = part + "_no_patch"
            return Path(*parts)
    return None


def _load_global(p: Path) -> Optional[torch.Tensor]:
    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        feat = data["feats"] if isinstance(data, dict) else data
        feat = feat.float()
        return feat if feat.dim() == 2 else feat.unsqueeze(0)
    except Exception:
        return None


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2):
        self.s = float(strength)
        self._gbank: Optional[Dict[int, torch.Tensor]] = None    # grade -> [n_roi, D] global-view bank
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            path = _path_of(pool, i)
            if path is None:
                continue
            gp = _no_patch_path(path)
            if gp is None or not gp.exists():
                continue
            g = _load_global(gp)
            if g is not None:
                chunks[int(round(float(pool[i][1])))].append(g)
        self._gbank = {k: torch.cat(v, dim=0) for k, v in chunks.items() if v}

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            bank = self._gbank.get(int(round(float(label)))) if self._gbank else None
            if bank is not None and bank.shape[0] >= 1 and bank.shape[1] == features.shape[1]:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
