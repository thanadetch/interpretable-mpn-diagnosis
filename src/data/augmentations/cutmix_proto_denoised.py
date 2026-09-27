"""cutmix_proto_denoised - prototype CutMix with PCA-denoised donor patches (train-only).

cutmix_proto_pull collapsed donors all the way onto the grade centroid (rank-1) and over-concentrated,
losing test. This variant denoises donors more gently: it projects each sampled near-centroid donor onto
the grade's top-R principal subspace (mean + reconstruction over the R dominant PCA directions of that
grade's patch bank). That keeps the dominant, reproducible grade structure while stripping the
per-patch high-frequency noise / outlier directions — cleaner than a raw donor, but far richer than the
single centroid direction. A middle point on the donor-purity axis that pull/shrink over-shot.

    donor* = mean_g + (donor - mean_g) @ V_R @ V_R^T ,   V_R = top-R PCA dirs of grade g's bank
    then CutMix round(strength*N) patches with donor*.  Label preserved.

`strength` = cutmix fraction. R = number of retained PCA components (module constant). Requires the train
pool. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

TAU = 0.4
R = 32       # retained PCA components per grade (denoising rank)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._w: Optional[Dict[int, torch.Tensor]] = None
        self._mean: Optional[Dict[int, torch.Tensor]] = None
        self._V: Optional[Dict[int, torch.Tensor]] = None
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
        bank, weights, means, Vs = {}, {}, {}, {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            c = allp.mean(dim=0, keepdim=True)
            dist = torch.norm(allp - c, dim=1)
            weights[g] = torch.softmax(-dist / (dist.std().clamp(min=1e-6) * TAU), dim=0)
            bank[g] = allp
            means[g] = c
            r = int(min(R, allp.shape[0], allp.shape[1]))
            try:
                _, _, Vh = torch.linalg.svd(allp - c, full_matrices=False)
                Vs[g] = Vh[:r].contiguous()          # [r, D]
            except Exception:
                Vs[g] = None
        self._bank, self._w, self._mean, self._V = bank, weights, means, Vs

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        bank = self._bank.get(g); w = self._w.get(g); c = self._mean.get(g); V = self._V.get(g)
        if bank is None or bank.shape[0] < 1:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        donors = bank[src].to(features.device, features.dtype)     # [k, D]
        if V is not None:
            Vd = V.to(features.device, features.dtype)             # [r, D]
            cc = c.to(features.device, features.dtype)
            donors = cc + ((donors - cc) @ Vd.t()) @ Vd            # denoise to top-R subspace
        out = features.clone()
        out[dst] = donors
        return out, float(label)
