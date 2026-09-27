"""stack_gv_anneal - champion CutMix + global-view tokens + ANNEALED schedule (train-only).

Every composition tried so far has exactly TWO stages (including the champion itself, cutmix + noise).
The 2026-07-25/26 rounds produced several mechanisms that each close a different part of the gap without
clearing both gates, and they fail in DIFFERENT directions - which is the same situation that produced the
champion. This is the first three-way stack:

    1. uniform within-grade CutMix at an ANNEALED strength  (0.8 -> 0 by ANNEAL_EPOCHS; the schedule
       direction that won its axis: start synthetic, finish on real bags)
    2. replace GV_FRAC of the bag with whole-ROI (global-view) tokens of same-grade TRAIN ROIs, on the
       SAME annealed schedule (so the extra magnification also disappears by the end)
    3. feature_noise @ 0.05

Rationale for stacking these two specifically: annealing was the only schedule that lifted the stubborn
titan/ASGAP test above its no-aug baseline (0.9622 > 0.9600) and gave v2/ASGAP val 0.8385; the global-view
dose was the only thing that held a champion-level TEST plateau on v2/ABMIL (0.9685-0.9688). Their
weaknesses are opposite (anneal loses test on ABMIL cells, global-view loses val everywhere), so the stack
tests whether the two deficits cancel.

Epoch is derived from the call count (epoch = calls // len(train_pool)); no trainer signal needed.
`strength` = initial cutmix fraction. No new data, no trainer edits. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

ANNEAL_EPOCHS = 30
GV_FRAC = 0.10
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


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._gbank: Optional[Dict[int, torch.Tensor]] = None
        self._n = 1
        self._calls = 0
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        self._n = max(1, len(pool))
        patches: Dict[int, list] = defaultdict(list)
        globals_: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            patches[g].append(item[0].float())
            path = _path_of(pool, i)
            gp = _no_patch_path(path) if path is not None else None
            if gp is not None and gp.exists():
                try:
                    data = torch.load(gp, map_location="cpu", weights_only=False)
                    vec = data["feats"] if isinstance(data, dict) else data
                    globals_[g].append(vec.float().reshape(1, -1))
                except Exception:
                    pass
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
        epoch = self._calls // self._n
        self._calls += 1
        decay = max(0.0, 1.0 - epoch / float(ANNEAL_EPOCHS))
        g = int(round(float(label)))
        n = features.shape[0]
        out = features
        s_t = self.s * decay
        bank = self._bank.get(g) if self._bank else None
        if s_t > 0.0 and bank is not None and bank.shape[0] >= 1:
            k = min(max(1, int(round(s_t * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        gv_t = GV_FRAC * decay
        gbank = self._gbank.get(g) if self._gbank else None
        if gv_t > 0.0 and gbank is not None and gbank.shape[0] >= 1 and gbank.shape[1] == features.shape[1]:
            kg = max(1, int(round(gv_t * n)))
            if kg >= 1:
                dstg = torch.randperm(n)[:min(kg, n)]
                srcg = torch.randint(gbank.shape[0], (dstg.numel(),))
                if out is features:
                    out = features.clone()
                out[dstg] = gbank[srcg].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
