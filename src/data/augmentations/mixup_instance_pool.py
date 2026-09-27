"""mixup_instance_pool - Ordinal instance-pool MixUp (feature-space, train-only).

Builds a mixed bag by taking a lambda-fraction of the CURRENT bag's patches and a
(1-lambda)-fraction of a randomly-sampled PARTNER bag's patches, then sets the regression
target to the patch-count-weighted blend of the two grades:

    lam ~ Beta(strength, strength)
    B~  = {n_a patches of A} U {n_b patches of B},  n_a~lam*N_a, n_b~(1-lam)*N_b
    y~  = p*y_a + (1-p)*y_b ,   p = n_a / (n_a + n_b)   (actual patch proportion)

Fits the scalar-regression + SmoothL1 formulation exactly (continuous target, no loss
change) and is pathology-meaningful: a bag that is part low-density and part high-density
marrow tissue genuinely has an intermediate diffuse fibrosis density.

`strength` = Beta concentration. small (->0): gentle (lam near 0/1, bags near-pure);
large (->inf): aggressive (lam near 0.5, always ~50/50 mixes). <=0 disables.

Permutation/size-invariant (operates on an unordered patch set, normalises by count),
deterministic given the global seed (uses torch's seeded RNG), MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.2):
        self.strength = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.strength <= 0.0 or pool is None or len(pool) < 2:
            return features, float(label)
        lam = float(torch.distributions.Beta(self.strength, self.strength).sample())
        j = int(torch.randint(len(pool), (1,)).item())
        partner = pool[j]
        feat_b = partner[0].to(features.device)
        label_b = float(partner[1])
        n_a = max(1, int(round(lam * features.shape[0])))
        n_b = max(1, int(round((1.0 - lam) * feat_b.shape[0])))
        idx_a = torch.randperm(features.shape[0], device=features.device)[:n_a]
        idx_b = torch.randperm(feat_b.shape[0], device=features.device)[:n_b]
        mixed = torch.cat([features[idx_a], feat_b[idx_b]], dim=0)
        p = n_a / (n_a + n_b)  # actual patch-proportion -> consistent mixed target
        return mixed, p * float(label) + (1.0 - p) * label_b
