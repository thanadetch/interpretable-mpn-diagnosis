"""cutmix_then_noise_tgtjitter35 - champion recipe + TARGET-side jitter (train-only, regression).

Every one of the existing feature-space augmentations keeps the regression TARGET fixed (or interpolates it
along a grade line). This adds a genuinely different, target-side regulariser on top of the champion recipe:
after uniform within-grade CutMix + isotropic feature noise, the returned regression target is jittered by
small Gaussian noise, target <- clip(g + N(0, TGTSIGMA^2), 0, 3). Motivation: MF reticulin grading has real
inter-observer variability (pathologists disagree by ~a grade), so a fuzzy target near g is a faithful label-
noise model — and because it regularises the LABEL not the feature distribution, it may decouple from the
val<->test anti-correlation trap that every feature-space lever hits. `strength` = cutmix fraction;
TGTSIGMA = target-noise std in grade units (per-file variant). MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

NOISE_SCALE = 0.05
TGTSIGMA = 0.35


class Augmentation(BaseAugmentation):
    requires_regression = True

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
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        # step 1: uniform within-grade cutmix
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
        # step 2: isotropic feature noise
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (NOISE_SCALE * std)
        # step 3: target-side jitter (label-noise model of inter-observer variability)
        tgt = float(label) + float(torch.randn(1).item()) * TGTSIGMA
        tgt = max(0.0, min(3.0, tgt))
        return out, tgt
