"""cutmix_soft - within-grade CutMix but BLEND donor with original (soft swap, train-only).

Standard cutmix hard-replaces patches; this blends: out[dst] = (1-m)*orig + m*donor, m=0.5.
Softer mixing keeps some of each real patch. Donors uniform from same-grade bank.
`strength` = fraction of patches blended. Requires train pool. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch
from . import BaseAugmentation
KWARGS = dict(strength=0.8)
MIX = 0.5
class Augmentation(BaseAugmentation):
    requires_regression = False
    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s=float(strength); self.max_bank=int(max_bank); self._bank=None; self._tried=False
    def _ensure(self, pool):
        if self._tried: return
        self._tried=True
        ch=defaultdict(list)
        for i in range(len(pool)):
            it=pool[i]; ch[int(round(float(it[1])))].append(it[0].float())
        b={}
        for g,l in ch.items():
            a=torch.cat(l,0)
            if a.shape[0]>self.max_bank: a=a[torch.randperm(a.shape[0])[:self.max_bank]]
            b[g]=a
        self._bank=b
    def __call__(self, features, label, pool=None):
        if self.s<=0.0 or pool is None or features.shape[0]<4: return features, float(label)
        self._ensure(pool)
        if self._bank is None: return features, float(label)
        g=int(round(float(label))); bank=self._bank.get(g)
        if bank is None or bank.shape[0]<1: return features, float(label)
        n=features.shape[0]; k=min(max(1,int(round(self.s*n))),n)
        dst=torch.randperm(n)[:k]; src=torch.randint(bank.shape[0],(k,))
        out=features.clone()
        out[dst]=(1-MIX)*out[dst]+MIX*bank[src].to(features.device,features.dtype)
        return out, float(label)
