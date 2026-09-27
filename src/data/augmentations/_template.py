"""_template - copy me to add a new feature-space augmentation.

    1.  cp _template.py my_aug.py
    2.  edit KWARGS defaults + the __call__ body
    3.  train with:   --augmentation my_aug [--aug_strength <float>]
    (no trainer edits needed - the registry auto-loads by filename)

CONTRACT
    __call__(features [N, D] already on the model's device,
             label    float grade of this bag,
             pool     the train Subset (indexable -> (feat, label, ...)) or None)
        -> (features_out [M, D], target_float)

    * Keep it permutation- and bag-size-invariant.
    * Deterministic given the global seed: use ONLY torch's seeded RNG
      (torch.rand / torch.randperm / torch.randint / torch.distributions),
      never Python's random or numpy without seeding.
    * Train-only: the trainer never calls augmentations on val/test.
    * Set requires_regression=False only if __call__ never changes the target.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.0)


class Augmentation(BaseAugmentation):
    requires_regression = True  # set False if it never changes `label`

    def __init__(self, strength: float = 0.0):
        self.strength = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.strength <= 0.0:
            return features, float(label)
        # TODO: your augmentation here. Building blocks you can use:
        #   - instance dropout:  keep = torch.rand(features.shape[0], device=features.device) > self.strength
        #                         features = features[keep] if keep.any() else features
        #   - seeded noise:       features = features + self.strength * torch.randn_like(features)
        #   - sample a partner:   j = int(torch.randint(len(pool), (1,)).item()); fb, yb = pool[j][0], pool[j][1]
        return features, float(label)
