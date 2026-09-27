"""cutmix_mid_protect - within-grade discrete patch transplant applied to the MIDDLE grades
(G1, G2) ONLY, leaving the extreme grades (G0, G3) pristine (train-only). NEW mechanism: a
grade-selective cutmix that directly targets why cutmix_within_grade fails the val gate.

cutmix_within_grade@0.7 reached the test+G1 corner (test 0.9663, G1 79.6) but lost val (0.7834) -
because it recombines patches in EVERY grade, including the 2-patient-per-grade extremes (G0/G3)
that drive the val_qwk on the tiny val cohort. This variant fires the SAME discrete same-grade
patch transplant but ONLY on G1 and G2 bags, returning G0 and G3 bags completely unchanged. The
hypothesis: keep the middle-grade diversity that gives cutmix its test+G1 strength, while protecting
the extreme-grade representations that the val cohort is sensitive to - the one registry-expressible
attempt at the val gate that does not need the (banned) sampler.

Distinct from cutmix_within_grade (which augments all grades): the grade-selective application is
the mechanism, motivated by the observed val-failure being an extreme-grade effect.

`strength` = fraction of a (G1/G2) bag's patches replaced (0 disables). Requires the train pool
(per-grade patch banks cached once, capped at ``max_bank``). Permutation/size-invariant, MPS-safe
(indexing only), no new deps.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.7)

_MIDDLE_GRADES = (1, 2)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.7, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None  # grade -> [M, D] (cpu), middle only
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            if g in _MIDDLE_GRADES:
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
        g = int(round(float(label)))
        if g not in _MIDDLE_GRADES:  # extremes (G0/G3) left pristine
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        bank = self._bank.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = max(1, int(round(self.s * n)))
        k = min(k, n)
        dst = torch.randperm(n)[:k]
        src = torch.randint(bank.shape[0], (k,))
        new = features.clone()
        new[dst] = bank[src].to(features.device, features.dtype)
        return new, float(label)
