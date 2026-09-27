"""cutmix_proto_gradeadaptive - prototype CutMix whose strength adapts to grade rarity (train-only).

Every CutMix variant so far used the same cutmix fraction for all grades. But the grades are imbalanced and
the minority grades (notably MF-1 / G1, the clinically hardest class) have the FEWEST real patches to begin
with — replacing 80% of a rare-grade bag with bank donors throws away most of its already-scarce genuine
evidence. This variant makes the cutmix fraction proportional to grade frequency: common grades get the
champion's aggressive fraction (strong regularisation where data is plentiful), rare grades get a gentler
fraction (preserve their scarce real patches, protecting minority-class recall).

    s_g = S_MIN + (strength - S_MIN) * (n_g / n_max)        n_g = #patches of grade g in the train pool
    replace round(s_g * N) patches with near-centroid (TAU=0.4) same-grade donors.  Label preserved.

`strength` = max (common-grade) cutmix fraction. Requires the train pool. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
S_MIN = 0.4     # cutmix fraction for the rarest grade


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._sg: Optional[Dict[int, float]] = None
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
        bank, weights, counts = {}, {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            counts[g] = allp.shape[0]
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
        n_max = max(counts.values()) if counts else 1
        self._sg = {g: S_MIN + (self.s - S_MIN) * (counts[g] / n_max) for g in counts}
        self._bank, self._w = bank, weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        s = float(self._sg.get(g, self.s))
        n = features.shape[0]
        k = min(max(1, int(round(s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
