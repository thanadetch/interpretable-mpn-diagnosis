"""multimix - instance-pool MixUp of K=3 bags (train-only). NEW: multi-bag (prior mixup = 2 bags).

Samples the current bag + (K-1) random partner bags, draws Dirichlet(strength) weights w over
the K bags, builds a merged bag by taking a w_k-fraction of each bag's patches, and sets the
regression target to the patch-count-weighted blend of the K grades. A higher-order extension
of ordinal instance-pool mixup that manufactures more diverse intermediate-density bags.

`strength` = Dirichlet concentration (higher -> more uniform 3-way mixes; lower -> closer to a
single bag). 0 disables. Fits scalar-regression + SmoothL1 (continuous target, no loss change).
Permutation/size-invariant, deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=1.0)
_K = 3


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 1.0):
        self.alpha = float(strength)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.alpha <= 0.0 or pool is None or len(pool) < _K:
            return features, float(label)
        feats = [features]
        labels = [float(label)]
        for _ in range(_K - 1):
            j = int(torch.randint(len(pool), (1,)).item())
            cand = pool[j]
            feats.append(cand[0].to(features.device))
            labels.append(float(cand[1]))
        w = torch.distributions.Dirichlet(torch.full((_K,), self.alpha)).sample()
        parts, ns = [], []
        for k in range(_K):
            nk = max(1, int(round(float(w[k]) * feats[k].shape[0])))
            idx = torch.randperm(feats[k].shape[0], device=features.device)[:nk]
            parts.append(feats[k][idx])
            ns.append(nk)
        mixed = torch.cat(parts, dim=0)
        tot = sum(ns)
        target = sum((ns[k] / tot) * labels[k] for k in range(_K))
        return mixed, float(target)
