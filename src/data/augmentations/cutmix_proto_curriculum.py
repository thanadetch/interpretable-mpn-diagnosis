"""cutmix_proto_curriculum - prototype CutMix with a strength that ramps up over training (train-only).

Every strength sweep here picked ONE fixed cutmix fraction for the whole run. This variant instead
anneals the fraction: it starts gentle (S0) so the model first learns clean grade features from
near-original bags (protecting early convergence / val), then ramps to the champion's aggressive
fraction (S1 = `strength`) so late training gets the strong regularisation that lifted test. The hope
is to get BOTH the high-val of light aug and the high-test of heavy aug, instead of trading one for
the other along the val<->test axis.

The augmentation object is instantiated once per run and reused for every bag, so a persistent call
counter estimates the epoch (calls_per_epoch ~= #train bags). Strength ramps linearly over the first
RAMP_EPOCHS, then holds at S1.

    epoch_est = calls // CALLS_PER_EPOCH
    s = S0 + (S1-S0) * min(1, epoch_est / RAMP_EPOCHS)      # then prototype CutMix at fraction s

`strength` = final (max) cutmix fraction S1. TAU=0.4 donors (champion). Requires the train pool.
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
S0 = 0.3                # starting cutmix fraction
RAMP_EPOCHS = 20        # ramp S0 -> S1 over this many (estimated) epochs
CALLS_PER_EPOCH = 857   # ~= number of train bags (patient split), for epoch estimation


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s1 = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._tried = False
        self._calls = 0

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            chunks[g].append(item[0].float())
        bank, weights = {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
        self._bank, self._w = bank, weights

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s1 <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        epoch_est = self._calls / CALLS_PER_EPOCH
        self._calls += 1
        frac = min(1.0, epoch_est / RAMP_EPOCHS)
        s = S0 + (self.s1 - S0) * frac
        n = features.shape[0]
        k = min(max(1, int(round(s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        return out, float(label)
