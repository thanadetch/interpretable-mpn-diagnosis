"""psemixp - PseMix following the ORIGINAL implementation (liupei101/PseMix).

Written against `utils/core.py` and `config/cfg_clf_mix.yaml` rather than a prose summary,
which is why the defaults and several mechanics differ from the earlier `psemix` module here.

ORIGINAL DEFAULTS (config/cfg_clf_mix.yaml), all adopted:

    pseb_dividing: proto      pseb_clustering: ProtoDiv    pseb_proto: mean
    pseb_n: 30                pseb_l: 8                    pseb_pheno_cut: uniform
    pseb_iter_tuning: 8       pseb_mixup_prob: 0.4         mixup_alpha: 1.0
    mixup_lam_from: content

MECHANISM

  (1) DIVIDE (ProtoDiv). prototype = mean instance; each instance scored by its distance to
      the prototype; the score range is cut into `l` phenotypes with UNIFORM (equal-width)
      bins; the phenotype assignment is then refined for `iter_tuning` rounds by recomputing
      phenotype centroids and reassigning instances to the nearest one. Finally each phenotype
      is spread over the `n` pseudo-bags by the original's `uniform_assign`:

          L = randperm(N) % num_label ; rlab = randperm(num_label) ; res = rlab[L]

  (2) DRAW lam ~ Beta(alpha, alpha), then discretise the ORIGINAL's way:

          lam_temp = int(lam * (n_pseb + 1))        # an integer in [0, n]

      lam_temp pseudo-bags are fetched from this bag.

  (3) MIX with probability `prob_mixup` (0.4): the remaining (n - lam_temp) pseudo-bags are
      taken from a partner bag and concatenated. Otherwise NO mixing happens and the bag is
      left as the lam_temp-pseudo-bag SUBSET - the original's "simple pseudo-bag sampling",
      which is a subsampling augmentation in its own right, not a no-op.

  (4) TARGET, `mixup_lam_from: content` (the default): ratio = lam_temp / n_pseb, i.e. the
      pseudo-bag COUNT ratio, not the instance-area ratio.

NOTE ON SCALE: n = 30 pseudo-bags is the original's default for slide-level bags of thousands
of instances. ROI bags here hold ~44 instances (min 13), so a pseudo-bag is 1-2 instances and
some come out empty. That is a genuine property of transplanting the method to this scale, not
an implementation fault, so the default is kept and empty pseudo-bags are simply skipped.

DEVIATIONS (honest):
  - FEATURE-LEVEL: cached foundation-model features, not raw tiles (the encoder is frozen).
  - Distance to prototype is cosine; the original's exact metric for ProtoDiv is not visible
    in the fetched excerpt.
  - The DIEM clustering alternative is not implemented (ProtoDiv is the config default).

Deterministic given the global torch seed; MPS-safe (division done on CPU-side ops only).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from . import BaseAugmentation

KWARGS = dict(strength=1.0)  # strength maps to the Beta alpha (original default 1.0)


def _uniform_assign(n: int, num_label: int, device) -> torch.Tensor:
    """The original's uniform_assign: balanced random labelling of n items into num_label groups."""
    L = torch.randperm(n, device=device) % num_label
    rlab = torch.randperm(num_label, device=device)
    return rlab[L]


class Augmentation(BaseAugmentation):
    requires_regression = True  # cross-class mixing produces fractional targets

    def __init__(self, strength: float = 1.0, n: int = 30, l: int = 8,
                 prob_mixup: float = 0.4, iter_tuning: int = 8,
                 pheno_cut: str = "uniform", lam_from: str = "content"):
        self.alpha = float(strength)
        self.n = int(n)
        self.l = int(l)
        self.prob_mixup = float(prob_mixup)
        self.iter_tuning = int(iter_tuning)
        self.pheno_cut = str(pheno_cut)
        self.lam_from = str(lam_from)

    # ── ProtoDiv division ────────────────────────────────────────────────
    def _divide(self, x: torch.Tensor) -> torch.Tensor:
        """Return a pseudo-bag index in [0, n) for every instance."""
        N = x.shape[0]
        xf = x.float()
        proto = xf.mean(dim=0, keepdim=True)
        score = 1.0 - F.cosine_similarity(xf, proto, dim=1)      # [N]

        if self.pheno_cut == "quantile":
            order = torch.argsort(score)
            pheno = torch.empty(N, dtype=torch.long, device=x.device)
            base, rem = divmod(N, self.l)
            start = 0
            for i in range(self.l):
                size = base + (1 if i < rem else 0)
                pheno[order[start:start + size]] = i
                start += size
        else:                                                     # uniform (the default)
            lo, hi = float(score.min()), float(score.max())
            step = (hi - lo) / self.l if hi > lo else 1.0
            pheno = torch.clamp(((score - lo) / step).long(), max=self.l - 1)

        # iterative refinement: recompute phenotype centroids, reassign to the nearest
        for _ in range(self.iter_tuning):
            cents, ids = [], []
            for k in range(self.l):
                m = pheno == k
                if int(m.sum()) > 0:
                    cents.append(xf[m].mean(dim=0))
                    ids.append(k)
            if len(cents) < 2:
                break
            C = torch.stack(cents, dim=0)
            d = (xf * xf).sum(1, keepdim=True) - 2.0 * (xf @ C.t()) + (C * C).sum(1)
            new = torch.tensor(ids, device=x.device)[d.argmin(dim=1)]
            if torch.equal(new, pheno):
                break
            pheno = new

        # spread each phenotype across the n pseudo-bags
        ind = torch.zeros(N, dtype=torch.long, device=x.device)
        for k in pheno.unique():
            m = pheno == k
            ind[m] = _uniform_assign(int(m.sum()), self.n, x.device)
        return ind

    @staticmethod
    def _fetch(x: torch.Tensor, ind: torch.Tensor, n: int, n_parts: int) -> Optional[torch.Tensor]:
        """The original's fetch_pseudo_bags: concatenate n_parts randomly chosen pseudo-bags."""
        if n_parts <= 0:
            return None
        picks = torch.randperm(n, device=x.device)[:n_parts]
        parts = [x[ind == p] for p in picks]
        parts = [p for p in parts if p.shape[0] > 0]      # ROI bags leave some pseudo-bags empty
        return torch.cat(parts, dim=0) if parts else None

    # ── augmentation ─────────────────────────────────────────────────────
    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if features.shape[0] < 2:
            return features, float(label)

        lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample()) if self.alpha > 0 else 1.0
        lam_temp = lam - 1e-5 if lam == 1.0 else lam
        k_a = int(lam_temp * (self.n + 1))                      # integer in [0, n]

        ind_a = self._divide(features)
        bag_a = self._fetch(features, ind_a, self.n, k_a)

        do_mix = pool is not None and len(pool) >= 2 and float(torch.rand(1).item()) <= self.prob_mixup
        if not do_mix:
            # "simple pseudo-bag sampling": keep the subset, label unchanged
            return (bag_a if bag_a is not None else features), float(label)

        j = int(torch.randint(len(pool), (1,)).item())
        partner = pool[j]
        feat_b = partner[0].to(features.device, features.dtype)
        label_b = float(partner[1])
        if feat_b.shape[0] < 2:
            return (bag_a if bag_a is not None else features), float(label)

        ind_b = self._divide(feat_b)
        bag_b = self._fetch(feat_b, ind_b, self.n, self.n - k_a)

        parts = [b for b in (bag_a, bag_b) if b is not None]
        if not parts:
            return features, float(label)
        mixed = torch.cat(parts, dim=0)

        if self.lam_from == "area":
            n_a = 0 if bag_a is None else bag_a.shape[0]
            ratio = n_a / mixed.shape[0]
        else:                                                    # content (the default)
            ratio = k_a / self.n
        target = ratio * float(label) + (1.0 - ratio) * label_b
        return mixed, float(min(3.0, max(0.0, target)))
