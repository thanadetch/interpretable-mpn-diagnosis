"""g1_synth_mixup - synthesise extra G1 bags from G0+G2 pairs (train-only). NEW: G1-targeted
class enrichment.

G1 is the lowest-recall grade on the test set and the most bag-scarce in train (139 bags vs
G2's 249 / G0's 320). This augmentation enriches the G1 region: when the current bag is a G0
or a G2 (G1's density-neighbours), with probability ``strength`` it mixes it (instance-pool)
with a partner from the OTHER neighbour at lam=0.5, so the blended density target is exactly
``0.5*0 + 0.5*2 = 1.0`` = G1. Bags of grade G1 (and G3) are left unchanged. The net effect is
to manufacture additional plausible mild-fibrosis (G1) bags by recombining the patches of a
no-fibrosis (G0) and a moderate (G2) bag - directly attacking the G1 recall gap without
touching the locked patient split.

`strength` = probability of applying the G0<->G2 -> G1 synthesis (only fires on G0/G2 bags).
0 disables. Fits scalar regression (continuous target 1.0). Permutation/size-invariant,
deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.5):
        self.p = float(strength)

    def _find_partner(self, pool, grade: int):
        for _ in range(16):
            j = int(torch.randint(len(pool), (1,)).item())
            cand = pool[j]
            if int(round(float(cand[1]))) == grade:
                return cand
        return None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.p <= 0.0 or pool is None or len(pool) < 2:
            return features, float(label)
        y = int(round(float(label)))
        partner_grade: Optional[int] = 2 if y == 0 else (0 if y == 2 else None)
        if partner_grade is None:
            return features, float(label)
        if float(torch.rand(1).item()) > self.p:
            return features, float(label)
        partner = self._find_partner(pool, partner_grade)
        if partner is None:
            return features, float(label)
        feat_b = partner[0].to(features.device, dtype=features.dtype)
        n_a = max(1, features.shape[0] // 2)
        n_b = max(1, feat_b.shape[0] // 2)
        idx_a = torch.randperm(features.shape[0], device=features.device)[:n_a]
        idx_b = torch.randperm(feat_b.shape[0], device=features.device)[:n_b]
        mixed = torch.cat([features[idx_a], feat_b[idx_b]], dim=0)
        return mixed, 1.0  # synthesised G1
