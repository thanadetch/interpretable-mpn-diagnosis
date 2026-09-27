"""cutmix_noise_s070 - COMPOSITION of the two winning augmentations (train-only).

The registry sweep showed exactly two augmentations that beat baseline on the fibrosis-grading
cohort: cutmix_within_grade (discrete same-grade patch transplant, the robust winner on
virchow2/uni2) and feature_noise (the only winner on the cutmix-hostile cells). Every registry
augmentation is a SINGLE mechanism; this is the one untested composition - stack them:

    1. cutmix_within_grade @ ``strength``  (replace a fraction of patches with real same-grade
       patches from the pooled bank; label-locked, on-manifold recombination)
    2. feature_noise @ 0.05                (add small within-bag per-dim Gaussian jitter to the
       resulting bag; the known-good regularising noise level)

Hypothesis: the discrete patch-diversity of cutmix and the manifold-smoothing of small noise are
orthogonal regularisers, so composing them may push past what either reaches alone. If it also
fails to beat cutmix@0.8, the registry (single + this composition) is definitively exhausted.

`strength` = cutmix replacement fraction (0 disables cutmix; noise still applied at 0.05).
Requires the train pool (per-grade patch banks, cached once). Permutation/size-invariant,
deterministic given the global seed, MPS-safe, no new deps.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

NOISE_SCALE = 0.07


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
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
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                sel = torch.randperm(allp.shape[0])[: self.max_bank]
                allp = allp[sel]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        # --- step 1: cutmix_within_grade ---
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            if self._bank is not None:
                g = int(round(float(label)))
                bank = self._bank.get(g)
                if bank is not None and bank.shape[0] >= 1:
                    n = features.shape[0]
                    k = min(max(1, int(round(self.s * n))), n)
                    dst = torch.randperm(n)[:k]
                    src = torch.randint(bank.shape[0], (k,))
                    out = features.clone()
                    out[dst] = bank[src].to(features.device, features.dtype)
        # --- step 2: feature_noise @ 0.05 ---
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (NOISE_SCALE * std)
        return out, float(label)
