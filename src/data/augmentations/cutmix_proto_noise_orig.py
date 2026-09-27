"""cutmix_proto_noise_orig - prototype CutMix + noise ONLY on the kept (non-donor) patches (train-only).

Combines the strengths of the two study champions while avoiding their conflict:
  - cutmix_prototype_sharp: donors drawn near the grade prototype (clean grade signal) — but no
    regularisation noise.
  - cutmix_then_noise: adds noise, but the global noise DIRTIES the freshly transplanted donors
    (this is why prototype+global-noise failed: it cancels the clean-donor benefit).

Fix: transplant prototype-weighted donors (keep them PRISTINE), and add feature noise ONLY to the
patches that were NOT replaced (the kept originals). So the clean real donors provide diversity while
the noise regularises the rest — the two act on disjoint patch sets and no longer conflict.

  step 1: replace k = round(strength*N) patches with same-grade donors ~ softmax(-dist/centroid) (TAU=0.4).
  step 2: add h += std * NOISE * N(0,1) to the (N-k) kept patches only.  Label preserved.

`strength` = cutmix fraction. Noise fixed at NOISE. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
NOISE = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            chunks[g].append(item[0].float())
        bank: Dict[int, torch.Tensor] = {}
        weights: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
        self._bank = bank
        self._w = weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        perm = torch.randperm(n)
        dst = perm[:k]                                            # replaced (donor) patches
        keep = perm[k:]                                          # kept originals
        src = torch.multinomial(w, k, replacement=True)          # prototype-weighted clean donors
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)  # pristine transplant
        if keep.numel() > 0:
            std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
            out[keep] = out[keep] + torch.randn_like(out[keep]) * (std * NOISE)  # noise only on kept
        return out, float(label)
