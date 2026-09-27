"""cutmix_bg_match - severity-matched background CutMix (train-only), grading-grounded + flaw-fixed.

Fixes the severity-corruption flaw of cutmix_worst_preserve. Same grading rationale: reticulin grade is
set by the densest/worst regions, so we PRESERVE the high-severity grade-driver patches and only augment
the low-severity background. The fix: donors are drawn from the SAME grade's LOW-severity pool (matched
to the background being replaced) instead of near the grade centroid, so the swap injects real
cross-patient background diversity WITHOUT raising the bag's mean severity (grade preserved).

  1. fibrosis severity axis v = normalise(centroid_maxgrade - centroid_mingrade)  (data-derived).
  2. per grade, the "background pool" = the lowest-severity half of that grade's patches.
  3. for a grade-g bag: replace the bottom-`strength` (lowest-severity) patches with donors sampled
     uniformly from grade g's background pool. Top (1-strength) severity drivers untouched. Label kept.

`strength` = fraction of low-severity background replaced. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

BG_FRAC = 0.5  # per-grade fraction (lowest-severity) that constitutes the "background" donor pool


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bg: Optional[Dict[int, torch.Tensor]] = None   # per-grade low-severity donor pool
        self._axis: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            chunks[g].append(item[0].float())
        banks: Dict[int, torch.Tensor] = {}
        centroids: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            banks[g] = allp
            centroids[g] = allp.mean(dim=0)
        gs = sorted(centroids.keys())
        if len(gs) < 2:
            return
        axis = centroids[gs[-1]] - centroids[gs[0]]
        self._axis = axis / axis.norm().clamp(min=1e-6)
        # per-grade low-severity background donor pool
        bg: Dict[int, torch.Tensor] = {}
        for g, allp in banks.items():
            sev = allp @ self._axis
            m = max(1, int(round(allp.shape[0] * BG_FRAC)))
            low_idx = torch.topk(sev, m, largest=False).indices
            bg[g] = allp[low_idx]
        self._bg = bg

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bg is None or self._axis is None:
            return features, float(label)
        g = int(round(float(label)))
        donors = self._bg.get(g)
        if donors is None or donors.shape[0] < 1:
            return features, float(label)
        axis = self._axis.to(features.device, features.dtype)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n - 1)
        severity = features @ axis
        dst = torch.topk(severity, k, largest=False).indices          # lowest-severity background
        src = torch.randint(donors.shape[0], (k,))                    # severity-matched donors
        out = features.clone()
        out[dst] = donors[src].to(features.device, features.dtype)
        return out, float(label)
