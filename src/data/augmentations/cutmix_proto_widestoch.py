"""cutmix_proto_widestoch - prototype CutMix with a WIDE per-bag stochastic strength (train-only).

cutmix_proto_stoch (the current best search outcome) samples the cutmix fraction per bag from a narrow band
U(center-0.1, center+0.1) around the champion's 0.8. The curriculum experiment showed that SCHEDULING the
strength (clean->aggressive over epochs) is worse than proto_stoch's per-bag random mixing. So this variant
keeps the per-bag randomness but WIDENS the range to span the whole clean<->aggressive spectrum every epoch:
each bag independently draws a fraction from U(LO, HI). Light draws preserve near-original grade evidence
(the lever that gives high val); heavy draws give strong regularisation (the lever that gives high test).
Seeing both every epoch directly targets the val<->test tradeoff that a single fixed strength cannot.

    per bag:  s ~ U(LO, HI) ;  replace round(s*N) patches with near-centroid (TAU=0.4) donors.  Label kept.

`strength` unused for the range (fixed LO..HI) but still gates on/off (>0). Requires the train pool.
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
LO = 0.35
HI = 0.95


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
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
        bank, weights = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
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
        n = features.shape[0]
        s = float(torch.empty(1).uniform_(LO, HI).item())
        k = min(max(1, int(round(s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
