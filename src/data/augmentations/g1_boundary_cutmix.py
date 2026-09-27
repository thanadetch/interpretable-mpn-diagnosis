"""g1_boundary_cutmix - inject a few ADJACENT-grade (G0+G2) patches into G1 bags, label-locked
(train-only). NEW mechanism: discrete boundary-patch transplant into the middle grade ONLY,
designed to attack the G1-recall gate WITHOUT touching the extreme grades.

Fires ONLY on G1 bags. Replaces a SMALL ``strength`` fraction of a G1 bag's patches with REAL
patches drawn from G0 and G2 bags (the two grades adjacent to G1), keeping the target at 1.0. The
G1 bag stays majority-G1 (cap the swap at half the patches) but now contains a few genuine
low-density (G0) and high-density (G2) patches - teaching the head that a G1 reading can tolerate
some borderline patches, i.e. tightening the G1 decision region from both sides. Crucially, G0 and
G3 bags are NEVER modified (they are only donors), so the extreme-grade recall that every mass-shift
augmentation collapses on the val cohort is left untouched - a direct, registry-only attempt to lift
G1 without the val penalty.

Distinct from g1_synth_mixup (which BLENDS G0+G2 into a NEW synthetic G1 and relabels) and
prototype_interp_adj (centroid translation): here EXISTING G1 bags receive a few DISCRETE, real
adjacent-grade patches; nothing is blended, the label is locked at 1.0, and only G1 bags change.

`strength` = fraction of a G1 bag's patches replaced (capped at 0.5; 0 disables). Requires the
train pool (G0/G2 patch banks cached once, capped at ``max_bank`` patches/grade). Permutation/size-
invariant, MPS-safe (indexing only), no new deps.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None  # grade -> [M, D] (cpu), grades {0,2}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            if g in (0, 2):
                chunks[g].append(item[0].float())
        bank: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                sel = torch.randperm(allp.shape[0])[: self.max_bank]
                allp = allp[sel]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        if int(round(float(label))) != 1:  # G1 bags only
            return features, float(label)
        self._ensure(pool)
        if not self._bank:
            return features, float(label)
        donors = [b for g, b in self._bank.items() if b is not None and b.shape[0] > 0]
        if not donors:
            return features, float(label)
        bank = torch.cat(donors, dim=0)  # mixed G0+G2 boundary patches
        n = features.shape[0]
        k = max(1, int(round(self.s * n)))
        k = min(k, n // 2)  # keep the bag majority-G1
        if k < 1:
            return features, float(label)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        new = features.clone()
        new[dst] = bank[src].to(features.device, features.dtype)
        return new, 1.0
