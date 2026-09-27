"""remixp - ReMix following the ORIGINAL implementation (TencentAILabHealthcare/ReMix).

Rewritten against the reference code rather than a prose description of it, which is why it
differs from the earlier `remix` / `remix_slot_*` modules in this registry:

    def mix_aug(src_feats, tgt_feats, mode, rate, strength, shift):
        closest_idxs = np.argmin(cdist(src_feats, tgt_feats), axis=1)
        for ix in range(len(src_feats)):
            if rand <= rate: ...   # one branch per mode; 'joint' runs all four

MECHANISM

  (1) REDUCE. k-means the bag's N patches into C prototypes (paper default C = 8).

  (2) PAIR. Draw ONE target bag uniformly from the same class
      (`positive_idxs = argwhere(train_labels == train_labels[idx]); choice(positive_idxs)`)
      and reduce it the same way. Same-class only, so the label never changes.

  (3) MATCH. For every source prototype, find the NEAREST target prototype by Euclidean
      distance (`cdist` + `argmin`). This is ReMix's prototype correspondence - a per-pair
      nearest-neighbour match, not a global slot assignment.

  (4) MIX, PER PROTOTYPE, each with independent probability `rate`:
        replace     - auged[ix] = tgt[closest[ix]]                       (in place)
        append      - append tgt[closest[ix]]                            (bag grows)
        interpolate - append (1-s)*auged[ix] + s*tgt[closest[ix]]        (bag grows)
        cov         - append auged[ix] + s * shift[closest[ix]][random]  (bag grows)
        joint       - all four branches, evaluated independently
        none        - no mixing at all: the REDUCE step alone. This is the original's
                      `--mode None`, and it isolates what reduction costs on its own,
                      separately from any augmentation.
      The README recommends a lower `rate` (0.1-0.2) for `joint` than the 0.5 default.

  `rate` is the PER-PROTOTYPE application probability; `strength` is the interpolation /
  shift coefficient. They are separate knobs in the original and are kept separate here
  (`--aug_strength` maps to `strength`; `rate` is a module constant per variant).

DEVIATIONS / APPROXIMATIONS (honest):
  - FEATURE-LEVEL: cached foundation-model patch features, not raw tiles (the encoder is
    frozen in this project, so augmentation cannot live upstream of it).
  - SEMANTIC SHIFT BANK: the original loads a precomputed `train_bag_feats_shift_{C}.npy`
    and indexes it by the matched prototype, sampling one of 200 shift vectors. The file's
    exact construction is not visible in the reference snippet, so the bank here is built the
    natural way: for every training bag and prototype, the deviations of that prototype's own
    member patches from the centroid, pooled per prototype index and capped at 200 vectors.
    This reproduces "add a plausible within-phenotype deviation" but may not match the
    original file byte-for-byte.
  - k-means is a short fixed-iteration Lloyd's algorithm on CPU (no k-means++ seeding).

Deterministic given the global torch seed, MPS-safe (clustering and distances on CPU).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

MODES = ("none", "replace", "append", "interpolate", "cov", "joint")


def _kmeans(x: torch.Tensor, c: int, iters: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd k-means on CPU. Returns (centroids [c, D], assignment [N])."""
    n = x.shape[0]
    if n <= c:
        return x, torch.arange(n)
    centroids = x[torch.randperm(n)[:c]].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        d = (x * x).sum(1, keepdim=True) - 2.0 * (x @ centroids.t()) + (centroids * centroids).sum(1)
        assign = d.argmin(dim=1)
        new = centroids.clone()
        for k in range(c):
            m = assign == k
            if int(m.sum()) > 0:
                new[k] = x[m].mean(dim=0)
        if torch.allclose(new, centroids, atol=1e-6):
            centroids = new
            break
        centroids = new
    return centroids, assign


class Augmentation(BaseAugmentation):
    requires_regression = False  # same-class donor -> the grade label is untouched

    def __init__(self, strength: float = 0.5, C: int = 8, rate: float = 0.5,
                 mode: str = "replace", n_shift: int = 200):
        self.strength = float(strength)
        self.C = int(C)
        self.rate = float(rate)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.n_shift = int(n_shift)
        self._protos: Optional[Dict[int, List[torch.Tensor]]] = None   # grade -> [ [C,D], ... ]
        self._shift: Optional[torch.Tensor] = None                     # [C, <=n_shift, D]
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        per_grade: Dict[int, List[torch.Tensor]] = defaultdict(list)
        shifts: List[List[torch.Tensor]] = [[] for _ in range(self.C)]
        for i in range(len(pool)):
            feat = pool[i][0].float()
            cent, assign = _kmeans(feat, self.C)
            per_grade[int(round(float(pool[i][1])))].append(cent)
            # within-prototype deviations = the "semantic shift" directions
            for k in range(min(self.C, cent.shape[0])):
                m = assign == k
                if int(m.sum()) > 0 and len(shifts[k]) < self.n_shift:
                    take = feat[m] - cent[k]
                    shifts[k].append(take[: self.n_shift - len(shifts[k])])
        self._protos = dict(per_grade)
        banks = []
        for k in range(self.C):
            b = torch.cat(shifts[k], dim=0)[: self.n_shift] if shifts[k] else None
            banks.append(b)
        self._shift = banks

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if features.shape[0] < 2:
            return features, float(label)
        if self.mode == "none":
            # `--mode None`: reduce only, no donor bag and no mixing.
            src, _ = _kmeans(features.detach().float().cpu(), self.C)
            return src.to(features.device, features.dtype), float(label)
        if self.rate <= 0.0 or pool is None:
            return features, float(label)
        self._ensure(pool)
        g = int(round(float(label)))
        cands = (self._protos or {}).get(g)
        if not cands:
            return features, float(label)

        dev, dt = features.device, features.dtype
        src, _ = _kmeans(features.detach().float().cpu(), self.C)          # [P, D]
        tgt = cands[int(torch.randint(len(cands), (1,)).item())]           # same-class donor

        # nearest target prototype for each source prototype (the paper's cdist + argmin)
        d = (src * src).sum(1, keepdim=True) - 2.0 * (src @ tgt.t()) + (tgt * tgt).sum(1)
        closest = d.argmin(dim=1)

        auged = [src[i] for i in range(src.shape[0])]
        s = self.strength
        do = lambda: float(torch.rand(1).item()) <= self.rate
        joint = self.mode == "joint"

        for ix in range(src.shape[0]):
            c = int(closest[ix])
            if (joint or self.mode == "replace") and do():
                auged[ix] = tgt[c]
            if (joint or self.mode == "append") and do():
                auged.append(tgt[c])
            if (joint or self.mode == "interpolate") and do():
                auged.append((1.0 - s) * auged[ix] + s * tgt[c])
            if (joint or self.mode == "cov") and do():
                bank = self._shift[c] if c < len(self._shift) else None
                if bank is not None and bank.shape[0] > 0:
                    j = int(torch.randint(bank.shape[0], (1,)).item())
                    auged.append(auged[ix] + s * bank[j])

        return torch.stack(auged, dim=0).to(dev, dt), float(label)
