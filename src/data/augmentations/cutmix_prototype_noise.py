"""cutmix_prototype_noise - prototype-guided CutMix + feature noise (train-only, combines 2 winners).

Combines the two augmentations that worked: prototype-weighted donor cutmix (donors near the grade
centroid, TAU=0.4) followed by small Gaussian feature noise (0.05). Tests whether stacking the
prototype donor-quality gain with mild noise regularisation pushes past prototype-cutmix alone.

    1. cutmix with donors sampled by softmax(-dist_to_centroid / (std*TAU)), TAU=0.4
    2. feature_noise @ 0.05
`strength` = cutmix fraction. Requires train pool. MPS-safe, deterministic given seed.
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
        self.s=float(strength); self.max_bank=int(max_bank); self._bank=None; self._w=None; self._tried=False
    def _ensure(self, pool):
        if self._tried: return
        self._tried=True
        ch=defaultdict(list)
        for i in range(len(pool)):
            it=pool[i]; ch[int(round(float(it[1])))].append(it[0].float())
        b={}; w={}
        for g,l in ch.items():
            a=torch.cat(l,0)
            if a.shape[0]>self.max_bank: a=a[torch.randperm(a.shape[0])[:self.max_bank]]
            cen=a.mean(0,keepdim=True); dist=torch.norm(a-cen,dim=1)
            b[g]=a; w[g]=torch.softmax(-dist/(dist.std().clamp(min=1e-6)*TAU),0)
        self._bank=b; self._w=w
    def __call__(self, features, label, pool=None):
        out=features
        if self.s>0.0 and pool is not None and features.shape[0]>=4:
            self._ensure(pool)
            if self._bank is not None:
                g=int(round(float(label))); bank=self._bank.get(g); w=self._w.get(g)
                if bank is not None and bank.shape[0]>=1:
                    n=features.shape[0]; k=min(max(1,int(round(self.s*n))),n)
                    dst=torch.randperm(n)[:k]; src=torch.multinomial(w,k,replacement=True)
                    out=features.clone(); out[dst]=bank[src].to(features.device,features.dtype)
        if out.shape[0]>=2:
            std=out.std(0,keepdim=True); out=out+torch.randn_like(out)*(NOISE*std)
        return out, float(label)
