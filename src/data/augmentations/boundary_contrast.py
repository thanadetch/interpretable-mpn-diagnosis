"""boundary_contrast - manufacture FRACTIONAL-target bags on the G1 boundaries (train-only).
NEW: boundary-sharpening (sub-integer targets), not whole-bag mass-shifting.

Every prior G1 attempt shifted a bag toward an INTEGER grade (centroid / axis / G1 synthesis),
which moves prediction mass to the middle and trips the val gate. This instead sharpens the
DECISION BOUNDARIES around G1 by manufacturing bags with the fractional targets 0.5 (the G0|G1
boundary) and 1.5 (the G1|G2 boundary): for a G0 or G1 bag it instance-pool-mixes with an adjacent
partner at lam=0.5 so the blended target is exactly y+0.5 or y-0.5. Teaching the scalar regressor
explicit half-grade anchors at G1's edges aims to tighten where G1 starts/ends (recall) without
dragging whole bags to the G1 centroid.

`strength` = probability of applying the boundary-contrast mix (fires on G0/G1/G2 bags, picking an
adjacent partner straddling a G1 edge). 0 disables. Fits scalar regression (continuous target).
Permutation/size-invariant, deterministic given the seed, MPS-safe, no new deps.
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

    def _find(self, pool, grade: int):
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
        # choose a partner grade that straddles a G1 boundary: G0<->G1 (->0.5) or G1<->G2 (->1.5)
        if y == 0:
            partner_grade = 1
        elif y == 2:
            partner_grade = 1
        elif y == 1:
            partner_grade = 0 if float(torch.rand(1).item()) < 0.5 else 2
        else:
            return features, float(label)  # G3 untouched
        if float(torch.rand(1).item()) > self.p:
            return features, float(label)
        partner = self._find(pool, partner_grade)
        if partner is None:
            return features, float(label)
        feat_b = partner[0].to(features.device, dtype=features.dtype)
        n_a = max(1, features.shape[0] // 2)
        n_b = max(1, feat_b.shape[0] // 2)
        idx_a = torch.randperm(features.shape[0], device=features.device)[:n_a]
        idx_b = torch.randperm(feat_b.shape[0], device=features.device)[:n_b]
        mixed = torch.cat([features[idx_a], feat_b[idx_b]], dim=0)
        target = 0.5 * y + 0.5 * partner_grade  # 0.5 or 1.5 (a G1 boundary)
        return mixed, float(target)
