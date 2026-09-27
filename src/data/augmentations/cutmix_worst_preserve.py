"""cutmix_worst_preserve - grade-driver-preserving CutMix (train-only), designed from the grading task.

Reticulin fibrosis grade (MF-0..3) is set by the DENSEST / worst-severity regions of the ROI; the
low-severity background merely dilutes them and varies between samplings without changing the grade.
So, unlike the other cutmix variants which replace RANDOM patches, this augmentation PRESERVES the
high-severity grade-driver patches and only augments the low-severity background:

  1. score each patch by fibrosis severity = projection onto the data-derived fibrosis axis
     v = normalise(centroid_maxgrade - centroid_mingrade)   (allowed data axis, no zero-shot labels).
  2. replace the BOTTOM-`strength` fraction (lowest severity = background) with real same-grade donors
     drawn near the grade prototype (softmax(-dist/centroid), TAU=0.4) — the top (1-strength) severity
     driver patches are left untouched.  Label preserved (same grade).

Rationale vs failed variants: keeps the grade signal intact (respects "grade = worst region"), while
injecting real cross-patient same-grade diversity into the dilutable background (fights 30-patient
overfitting). `strength` = fraction of (lowest-severity) patches replaced. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

TAU = 0.4  # donor concentration toward the grade prototype (matches cutmix_prototype_sharp)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._axis: Optional[torch.Tensor] = None       # [D] fibrosis severity direction
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
        bank: Dict[int, torch.Tensor] = {}
        weights: Dict[int, torch.Tensor] = {}
        centroids: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0)
            dist = torch.norm(allp - c.unsqueeze(0), dim=1)
            w = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            weights[g] = w
            centroids[g] = c
        # data-derived fibrosis severity axis: max-grade centroid - min-grade centroid
        gs = sorted(centroids.keys())
        if len(gs) >= 2:
            axis = centroids[gs[-1]] - centroids[gs[0]]
            self._axis = axis / axis.norm().clamp(min=1e-6)
        self._bank = bank
        self._w = weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None or self._axis is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        axis = self._axis.to(features.device, features.dtype)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n - 1)   # keep >=1 driver patch
        severity = features @ axis                        # [N] higher = more fibrotic
        dst = torch.topk(severity, k, largest=False).indices  # lowest-severity background patches
        src = torch.multinomial(w, k, replacement=True)   # clean same-grade prototype donors
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
