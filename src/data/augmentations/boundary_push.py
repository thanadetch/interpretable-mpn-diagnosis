"""boundary_push - manufacture near-boundary HARD examples (train-only).

A different philosophy from every CutMix / cleaning variant tried so far. Instead of making bags MORE
grade-typical (which is already at its optimum), this nudges a subset of patches a small, bounded fraction
TOWARD the nearest OTHER-grade centroid while KEEPING the true label. That moves those patches closer to
the inter-grade decision boundary without crossing it (the shift is a small fraction of the centroid gap),
producing harder positive examples that should sharpen the boundary and improve generalisation on the
ambiguous mid-grades (e.g. MF-1 vs MF-2).

    for a random subset (fraction `strength` of patches):
        c_own = own grade centroid ; c_near = nearest DIFFERENT-grade centroid
        a ~ U(0, ALPHA_MAX)
        h' = h + a * (c_near - c_own)          # bounded push toward the boundary, label unchanged

`strength` = fraction of patches pushed. ALPHA_MAX = max push as a fraction of the centroid gap (kept
small so patches stay clearly on their own side). Requires the train pool. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

ALPHA_MAX = 0.25    # max fraction of the (c_near - c_own) gap to travel


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._cent: Optional[Dict[int, torch.Tensor]] = None
        self._near: Optional[Dict[int, int]] = None
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
        cent: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cent[g] = allp.mean(dim=0, keepdim=True)      # [1, D]
        # nearest different-grade centroid for each grade
        near: Dict[int, int] = {}
        gs = list(cent.keys())
        for g in gs:
            best, bd = None, None
            for h in gs:
                if h == g:
                    continue
                d = torch.norm(cent[g] - cent[h]).item()
                if bd is None or d < bd:
                    bd, best = d, h
            near[g] = best if best is not None else g
        self._cent, self._near = cent, near

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._cent is None:
            return features, float(label)
        g = int(round(float(label)))
        c_own = self._cent.get(g)
        h_near = self._near.get(g)
        if c_own is None or h_near is None or h_near == g:
            return features, float(label)
        c_near = self._cent.get(h_near)
        if c_near is None:
            return features, float(label)
        direction = (c_near - c_own).to(features.device, features.dtype)   # [1, D]
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        a = torch.empty(k, 1, device=features.device, dtype=features.dtype).uniform_(0.0, ALPHA_MAX)
        out = features.clone()
        out[dst] = out[dst] + a * direction
        return out, float(label)
