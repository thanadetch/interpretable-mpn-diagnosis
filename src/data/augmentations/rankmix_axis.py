"""rankmix_axis - RankMix-INSPIRED rank-then-mix-salient augmentation (PROXY, train-only).

Paper: Yuan-Chih Chen & Cheng-Hung Lu, "RankMix: Data Augmentation for Weakly Supervised
Learning of Classifying Whole Slide Images with Diverse Sizes and Imbalanced Categories",
CVPR 2023.

MECHANISM (faithful to RankMix's *rank-then-mix-salient* idea): rank the instances of two WSIs by
a per-instance CONTRIBUTION score, take the TOP-ranked (salient) regions of each, and form a mixed
bag from a lam-weighted blend of the two salient sets; the regression target is the same lam-blend
of the two slide labels. A bag stays a full bag by keeping bag A's lower-ranked (non-salient) tail
unchanged and replacing only its salient head with the cross-bag salient mixture.

    tau = strength                      # top-ranked fraction treated as "salient"
    s_i = features_i . v                # contribution score (projection on fibrosis axis v)
    lam ~ Beta(1, 1) (uniform)          # ranked-mixup proportion, like RankMix
    head~ = {lam*k_a top of A's salient} U {(1-lam)*k_b top of B's salient}
    bag~  = head~  U  {A's lower-ranked (1-tau) tail}   # full bag, A-anchored
    y~    = clip( lam*y_a + (1-lam)*y_b , 0, 3 )

PROXY / DEVIATION (REQUIRED honesty): RankMix's original ranking uses model attention /
pseudo-label CONFIDENCE from a TRAINED MIL model. The augmentation registry runs
before/independently of the MIL model and has no access to it, so we substitute a MODEL-FREE
ranking signal: a data-derived fibrosis-axis projection score, v = normalize(mean(features of
bags with grade>=2) - mean(grade<=1)) -- the same grade axis used elsewhere in this repo. This
faithfully reproduces RankMix's rank-then-mix-salient mechanism but with a different (model-free)
ranking signal. It does NOT implement RankMix's iterative self-distillation / pseudo-label
re-ranking loop (that needs a trained model in the loop). Hence: a PROXY, not faithful RankMix.

`strength` = tau, the top-ranked salient fraction (in (0,1]). <=0 disables (no-op). Requires the
train pool (axis v computed once, cached; bag features sampled live per call -> nothing large is
cached). Fits scalar regression (soft target). Permutation/size-invariant (ranks an unordered
patch set), deterministic given the global seed (torch RNG only), MPS-safe (matmul/topk only),
no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)


class Augmentation(BaseAugmentation):
    requires_regression = True

    def __init__(self, strength: float = 0.5):
        self.tau = float(strength)
        self._v: Optional[torch.Tensor] = None  # fibrosis axis [D] (cpu)
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        means, grades = [], []
        for i in range(len(pool)):
            item = pool[i]
            means.append(item[0].float().mean(dim=0))
            grades.append(float(item[1]))
        if len(means) < 4:
            return
        M = torch.stack(means)            # [B, D]
        gt = torch.tensor(grades)         # [B]
        hi = gt >= 2.0
        lo = gt <= 1.0
        if int(hi.sum()) and int(lo.sum()):
            v = M[hi].mean(dim=0) - M[lo].mean(dim=0)
            v = v / (v.norm() + 1e-8)
            self._v = v

    @staticmethod
    def _rank_desc(feats: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return indices of ``feats`` sorted by contribution score (descending)."""
        s = feats @ v                     # [N] projection on fibrosis axis
        return torch.argsort(s, descending=True)

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.tau <= 0.0 or pool is None or len(pool) < 2 or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._v is None:
            return features, float(label)
        v = self._v.to(features.device, features.dtype)

        # sample a partner bag
        j = int(torch.randint(len(pool), (1,)).item())
        partner = pool[j]
        feat_b = partner[0].to(features.device, features.dtype)
        label_b = float(partner[1])
        if feat_b.shape[0] < 1:
            return features, float(label)

        tau = min(1.0, self.tau)
        n_a = features.shape[0]
        n_b = feat_b.shape[0]

        # rank each bag by the (proxy) contribution score, descending
        order_a = self._rank_desc(features, v)
        order_b = self._rank_desc(feat_b, v)

        # salient head sizes (top-tau of each bag)
        k_a = max(1, int(round(tau * n_a)))
        k_a = min(k_a, n_a)
        k_b = max(1, int(round(tau * n_b)))
        k_b = min(k_b, n_b)

        salient_a = features[order_a[:k_a]]   # A's top-ranked (salient) patches
        salient_b = feat_b[order_b[:k_b]]     # B's top-ranked (salient) patches
        tail_a = features[order_a[k_a:]]      # A's lower-ranked tail (keeps bag full, A-anchored)

        # ranked mixup proportion (lam ~ Beta(1,1) = uniform), like RankMix's ranked mix
        lam = float(torch.distributions.Beta(1.0, 1.0).sample())

        # take lam-fraction of A's salient head + (1-lam)-fraction of B's salient head
        m_a = max(1, int(round(lam * k_a)))
        m_a = min(m_a, k_a)
        m_b = max(1, int(round((1.0 - lam) * k_b)))
        m_b = min(m_b, k_b)
        head = torch.cat([salient_a[:m_a], salient_b[:m_b]], dim=0)

        mixed = torch.cat([head, tail_a], dim=0) if tail_a.shape[0] > 0 else head

        target = lam * float(label) + (1.0 - lam) * label_b
        return mixed, min(3.0, max(0.0, target))
