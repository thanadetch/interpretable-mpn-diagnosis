"""bag_mix_samegrade - CutMix patches from ONE real same-grade slide (train-only).

Every prior CutMix variant here (cutmix_within_grade, cutmix_prototype_sharp, cutmix_proto_stoch) draws
donor patches from a GLOBAL same-grade bank — patches pooled across many slides — which destroys the
within-slide co-occurrence structure of the transplanted tissue. This variant instead picks ONE random
other slide of the same grade and transplants a contiguous random subset of ITS patches, preserving the
donor's intra-slide correlation (the real spatial/textural context a single biopsy region carries).
That is the most faithful "MixUp of two real slides" and a distinct hypothesis from centroid-bank CutMix.

    per bag:  pick a random other same-grade bag D; k = round(strength*N);
              replace k random patches of this bag with k random patches drawn from D (no replacement if D big enough).

`strength` = fraction of patches replaced. Requires the train pool. Label preserved. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
        self.s = float(strength)
        self._bags: Optional[Dict[int, List[torch.Tensor]]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        bags: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            bags[g].append(item[0].float())
        self._bags = bags

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        g = int(round(float(label)))
        donors = self._bags.get(g) if self._bags else None
        if not donors or len(donors) < 2:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        # pick a random donor slide of the same grade (any; self-collision is harmless dither)
        d = donors[int(torch.randint(len(donors), (1,)).item())].to(features.device, features.dtype)
        if d.shape[0] == 0:
            return features, float(label)
        dst = torch.randperm(n)[:k]
        src = torch.randint(d.shape[0], (k,))
        out = features.clone()
        out[dst] = d[src]
        return out, float(label)
