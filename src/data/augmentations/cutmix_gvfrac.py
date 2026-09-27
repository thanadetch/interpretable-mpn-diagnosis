"""cutmix_gvfrac - champion CutMix + a SWEEPABLE dose of whole-ROI (global-view) tokens (train-only).

Same recipe as ``cutmix_then_globalview`` but with the knobs swapped so the new axis is the one the CLI
controls: CutMix is fixed at the champion's 0.8 and ``strength`` = the FRACTION of the bag turned into
whole-ROI (no_patch) tokens of the same grade. Motivated by the first global-view round:

  * 20% global tokens WITHOUT CutMix is toxic (v2/ABMIL 0.7067/0.9000) - the whole-ROI embedding is a
    different view and a bag cannot be mostly made of it;
  * 15% global tokens ON TOP of CutMix is neutral-to-tied (v2/ABMIL 0.8262/0.9688, the day's best ABMIL
    test) - so the dose, not the idea, is what the previous round actually measured;
  * global tokens lifted G1 recall specifically (titan/ABMIL 89.8, the highest G1 anywhere in the study),
    which is mechanistically plausible: G1 is defined by the CONTINUITY of the fibre network across the
    field, a global property no single patch can express - and G1 is this cohort's persistent weak class.

So this sweeps the dose to see whether a small amount of global context is an additive gain (and whether
the G1 effect survives at a dose that does not damage the rest).

`strength` = global-token fraction (0.05 / 0.10 / 0.25 ...); CUTMIX below is fixed at 0.8.
Donors come only from train ROIs, no leakage; no new data, no extraction, no trainer edits. MPS-safe.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.1)

CUTMIX = 0.8
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

    def __init__(self, strength: float = 0.1, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._gbank: Optional[Dict[int, torch.Tensor]] = None
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
        bank = self._bank.get(g) if self._bank else None
        if bank is not None and bank.shape[0] >= 1:
            k = min(max(1, int(round(CUTMIX * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        gbank = self._gbank.get(g) if self._gbank else None
        if self.s > 0.0 and gbank is not None and gbank.shape[0] >= 1 and gbank.shape[1] == features.shape[1]:
            kg = min(max(1, int(round(self.s * n))), n)
            dstg = torch.randperm(n)[:kg]
            srcg = torch.randint(gbank.shape[0], (kg,))
            if out is features:
                out = features.clone()
            out[dstg] = gbank[srcg].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
