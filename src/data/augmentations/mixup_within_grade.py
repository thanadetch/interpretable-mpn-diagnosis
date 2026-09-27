"""mixup_within_grade - instance-pool MixUp restricted to SAME-GRADE partner bags.

Like mixup_instance_pool, but the partner bag is sampled to have the SAME grade as the
current bag, so the mixed bag stays unambiguously that grade and the target is UNCHANGED
(no soft/interpolated label, hence NO label noise). Pure intra-grade variety augmentation:
it manufactures new plausible bags of a given grade by recombining patches of two real
bags of that grade.

`strength` = Beta(strength, strength) concentration controlling the A:B patch split.
0 disables. Safer than cross-grade mixup (avoids the label-noise risk of mixing G1 with G3).

Permutation/size-invariant, deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


class Augmentation(BaseAugmentation):
    requires_regression = False  # partner is same grade -> target unchanged

    def __init__(self, strength: float = 0.2):
        self.alpha = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.alpha <= 0.0 or pool is None or len(pool) < 2:
            return features, float(label)
        partner = None
        for _ in range(8):  # sample a same-grade partner (capped retries)
            j = int(torch.randint(len(pool), (1,)).item())
            cand = pool[j]
            if float(cand[1]) == float(label):
                partner = cand
                break
        if partner is None:
            return features, float(label)
        feat_b = partner[0].to(features.device)
        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())
        n_a = max(1, int(round(lam * features.shape[0])))
        n_b = max(1, int(round((1.0 - lam) * feat_b.shape[0])))
        idx_a = torch.randperm(features.shape[0], device=features.device)[:n_a]
        idx_b = torch.randperm(feat_b.shape[0], device=features.device)[:n_b]
        mixed = torch.cat([features[idx_a], feat_b[idx_b]], dim=0)
        return mixed, float(label)  # same grade -> target unchanged
