"""mixstyle_bag - cross-bag feature-statistics (style) mixing, content/label preserved (train-only).
NEW mechanism (Zhou et al., MixStyle 2021), motivated by this cohort's real SCALE HETEROGENEITY
(20/50/100/200µm ROIs) that no prior augmentation addressed.

For a bag A, compute its per-feature-dim mean muA and std sigmaA across patches (the bag's "style" /
first+second moments). Draw a random OTHER bag B from the train pool (style source) and a mixing
weight lam ~ Beta(s, s). Renormalise A's patches to a MIXED style:
    x_hat = sigma_mix * (x - muA) / (sigmaA + eps) + mu_mix
with mu_mix = lam*muA + (1-lam)*muB, sigma_mix = lam*sigmaA + (1-lam)*sigmaB. The per-patch CONTENT
(z-scored structure within the bag) is preserved and the GRADE/label is unchanged - only the bag's
global style moments are perturbed toward another bag's. This injects style/scale invariance, the
domain-generalisation prior MixStyle is built for, without mixing or swapping any actual patch
(distinct from mixup = convex blend, cutmix = discrete swap, feature_noise = additive jitter,
fibrosis_axis_shift = directional translate).

`strength` = the Beta(s, s) concentration (larger -> milder style shift near lam=0.5; smaller ->
stronger, near the donor's style). 0 disables. Requires the train pool (per-bag (mu, sigma) cached
once, ~light). Permutation/size-invariant, deterministic given the seed, MPS-safe (no eigh/cdist),
no new deps.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.1)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.1):
        self.s = float(strength)
        self._stats: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # [(mu[D], sigma[D])] cpu
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        stats = []
        for i in range(len(pool)):
            f = pool[i][0].float()
            mu = f.mean(dim=0)
            sigma = f.std(dim=0)
            stats.append((mu, sigma))
        if len(stats) >= 2:
            self._stats = stats

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 2:
            return features, float(label)
        self._ensure(pool)
        if self._stats is None:
            return features, float(label)
        dev, dt = features.device, features.dtype
        muA = features.mean(dim=0)
        sigmaA = features.std(dim=0)
        j = int(torch.randint(len(self._stats), (1,)).item())
        muB, sigmaB = self._stats[j]
        muB = muB.to(dev, dt)
        sigmaB = sigmaB.to(dev, dt)
        # Beta(s, s) mixing weight, sampled on CPU
        a = torch.tensor([self.s], dtype=torch.float32)
        lam = torch.distributions.Beta(a, a).sample().item()
        mu_mix = lam * muA + (1.0 - lam) * muB
        sigma_mix = lam * sigmaA + (1.0 - lam) * sigmaB
        eps = 1e-6
        x_hat = sigma_mix.unsqueeze(0) * (features - muA.unsqueeze(0)) / (sigmaA.unsqueeze(0) + eps) + mu_mix.unsqueeze(0)
        return x_hat, float(label)
