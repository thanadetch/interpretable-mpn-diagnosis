"""cutmix_then_globalview - champion CutMix, then a few WHOLE-ROI (global-view) tokens (train-only).

Composition of the established champion with the one new information source in the study (see
``cutmix_globalview_noise`` for why the whole-ROI embedding is not redundant with the patch view):

    1. uniform within-grade CutMix @ ``strength``          (the champion's stage 1, 0.8)
    2. replace GV_FRAC of the resulting patches with whole-ROI embeddings of same-grade TRAIN ROIs
    3. feature_noise @ 0.05                                (the champion's stage 2)

Rationale for composing rather than replacing: the 2026-07-25 rounds showed stage 1 carries the entire
gain, so the global-view tokens should be added ON TOP of it, not instead of it. If the composition beats
the champion, the gain is attributable to the added magnification level (a real contribution, since it is
new information, not a re-arrangement); if it does not, the patch-view ceiling is confirmed against an
information-adding mechanism - the strongest form of the exhaustion argument.

`strength` = cutmix fraction; GV_FRAC below = share of the bag turned into global-view tokens.
No new data / extraction / trainer edits. MPS-safe, deterministic given the global seed.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

GV_FRAC = 0.15
SIGMA = 0.05


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

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None      # patch bank
        self._gbank: Optional[Dict[int, torch.Tensor]] = None     # global-view bank
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        patches: Dict[int, list] = defaultdict(list)
        globals_: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            patches[g].append(item[0].float())
            path = _path_of(pool, i)
            gp = _no_patch_path(path) if path is not None else None
            if gp is not None and gp.exists():
                gv = _load_global(gp)
                if gv is not None:
                    globals_[g].append(gv)
        bank = {}
        for g, lst in patches.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank
        self._gbank = {k: torch.cat(v, dim=0) for k, v in globals_.items() if v}

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        g = int(round(float(label)))
        n = features.shape[0]
        out = features
        # --- step 1: uniform within-grade cutmix ---
        bank = self._bank.get(g) if self._bank else None
        if self.s > 0.0 and bank is not None and bank.shape[0] >= 1:
            k = min(max(1, int(round(self.s * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        # --- step 2: global-view tokens ---
        gbank = self._gbank.get(g) if self._gbank else None
        if gbank is not None and gbank.shape[0] >= 1 and gbank.shape[1] == features.shape[1]:
            kg = min(max(1, int(round(GV_FRAC * n))), n)
            dstg = torch.randperm(n)[:kg]
            srcg = torch.randint(gbank.shape[0], (kg,))
            if out is features:
                out = features.clone()
            out[dstg] = gbank[srcg].to(features.device, features.dtype)
        # --- step 3: feature noise ---
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
