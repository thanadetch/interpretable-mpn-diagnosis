"""cutmix_mean_preserve - prototype CutMix that preserves the bag's mean embedding (train-only).

CutMix collapses slide-level TITAN because TITAN reads a bag mostly through its aggregate (≈ mean) summary,
and replacing patches with cross-slide donors shifts that mean off the slide-level manifold. This variant
does the champion's prototype CutMix, then RE-CENTERS the whole bag so its mean embedding equals the
ORIGINAL bag mean. The per-patch composition is still diversified (helps the patch-level backbones
virchow2/uni2), but the slide-level summary is left unchanged (should spare TITAN the collapse). If it
works, it is the single augmentation that helps all three backbones — what the earlier search never found.

    1. prototype CutMix (TAU=0.4) at fraction `strength`  ->  out
    2. out <- out + (mean(features) - mean(out))          # restore original bag-mean embedding
    label preserved.

`strength` = cutmix fraction. Requires the train pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4


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
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        # restore the original bag-mean embedding (preserve the slide-level summary for TITAN)
        out = out + (features.mean(dim=0, keepdim=True) - out.mean(dim=0, keepdim=True))
        return out, float(label)
