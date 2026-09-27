"""cutmix_ramp_noise - champion CutMix RAMPED UP over training + noise (train-only).

The opposite schedule to ``cutmix_anneal_noise`` on the same untouched TIME axis: instead of removing the
augmentation at the end, introduce it gradually.

    strength_t = S_MIN + (``strength`` - S_MIN) * min(1, epoch / RAMP_EPOCHS)   (then feature_noise @ 0.05)

Rationale: at 80% replacement from step 1, the model never sees a clean bag while its features are still
being organised, which is the regime where several cells collapse to best-epoch 1-3 (titan especially).
Ramping lets the head fit real bags first and adds synthetic diversity once it is past the unstable phase -
progressive/curriculum augmentation in its simplest, mechanism-free form (the prototype-donor curriculum
that was tested earlier varied donor QUALITY, never the amount).

The ramp/anneal pair brackets the schedule axis: if neither beats the constant champion, augmentation
timing carries no signal here and the axis closes with two runs per cell. Epoch is derived from the call
count (epoch = calls // len(train_pool)); no trainer signal needed. `strength` = the final cutmix fraction.
MPS-safe, deterministic given the global seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

S_MIN = 0.2
RAMP_EPOCHS = 20
SIGMA = 0.05


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._n = 1
        self._calls = 0
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        self._n = max(1, len(pool))
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            chunks[int(round(float(item[1])))].append(item[0].float())
        bank = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        epoch = self._calls // self._n
        self._calls += 1
        frac = min(1.0, epoch / float(RAMP_EPOCHS))
        s_t = S_MIN + (self.s - S_MIN) * frac
        out = features
        if s_t > 0.0 and self._bank is not None:
            bank = self._bank.get(int(round(float(label))))
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(s_t * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
