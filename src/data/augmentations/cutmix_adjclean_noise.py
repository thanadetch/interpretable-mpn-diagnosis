"""cutmix_adjclean_noise - CutMix that replaces the ADJACENT-GRADE-LOOKING patches + noise (train-only).

Everything about the CutMix stage has been swept except ONE thing: WHICH patches of the recipient bag get
replaced. Every module in the registry uses ``dst = randperm(n)[:k]`` (uniform destinations); the only
destination rule ever tried was within-grade typicality (cutmix_replace_outliers, where random dst won).

This module picks destinations from CROSS-GRADE geometry instead, which is what the causal mean-readout
finding says the grade is actually read from: a real ROI contains regions of mixed severity, so the
patches that look like the NEIGHBOURING grade drag the bag mean off its grade position. Replacing exactly
those with typical same-grade tissue "cleans" the bag toward its own grade without touching the patches
that already agree with the label.

    score(patch) = max_{g' adjacent to g} cos(patch, c_g') - cos(patch, c_g)
    replace the top-``strength`` fraction by that score with uniform same-grade donors, then noise @0.05

Grade centroids c_g come from the train pool (cached once). If it loses to random destinations, the
destination axis is closed from the cross-grade side as well and the CutMix stage has no free parameter
left except bag cardinality (see cutmix_sizejitter_noise). MPS-safe, deterministic, no trainer edits.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._cent: Optional[Dict[int, torch.Tensor]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            chunks[int(round(float(item[1])))].append(item[0].float())
        bank, cent = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            c = allp.mean(dim=0)
            cent[g] = c / c.norm().clamp(min=1e-6)
        self._bank, self._cent = bank, cent

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            g = int(round(float(label)))
            bank = self._bank.get(g) if self._bank is not None else None
            if bank is not None and bank.shape[0] >= 1 and self._cent is not None and g in self._cent:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                fn = features / features.norm(dim=1, keepdim=True).clamp(min=1e-6)
                own = fn @ self._cent[g].to(features.device, features.dtype)           # [N]
                adj = None
                for gp in (g - 1, g + 1):
                    if gp in self._cent:
                        sim = fn @ self._cent[gp].to(features.device, features.dtype)
                        adj = sim if adj is None else torch.maximum(adj, sim)
                if adj is not None:
                    dst = torch.topk(adj - own, k).indices                             # most "off-grade"
                else:
                    dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
