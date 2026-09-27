"""cutmix_margin_noise - within-grade CutMix with MARGIN-selected (grade-confident) donors + noise (train-only).

Completes the donor-selection space. Prototype cutmix picks donors NEAR their own grade centroid; this instead
picks donors with high inter-grade MARGIN = (distance to the NEAREST OTHER-grade centroid) - (distance to own
grade centroid). High-margin patches sit deep in their own grade's territory, far from decision boundaries =
the most unambiguously-this-grade evidence. Sampled with softmax(margin/TAU) so higher-margin donors are
likelier. Then the champion's isotropic all-patch noise. Tests whether grade-CONFIDENT donors (a separability
criterion, orthogonal to the centrality criterion of prototype) help — expected to concentrate donors and thus
favour ABMIL over ASGAP per the diverse-donor finding, but genuinely untested. `strength` = cutmix fraction.
MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05
TAU = 0.5


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
        cents: Dict[int, torch.Tensor] = {}
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            cents[g] = allp.mean(dim=0)
        # margin weights: dist to nearest OTHER-grade centroid - dist to own centroid
        weights: Dict[int, torch.Tensor] = {}
        grades = sorted(cents.keys())
        for g in grades:
            allp = bank[g]
            d_own = torch.norm(allp - cents[g].unsqueeze(0), dim=1)
            others = [cents[h] for h in grades if h != g]
            if not others:
                weights[g] = torch.ones(allp.shape[0]) / allp.shape[0]
                continue
            O = torch.stack(others, dim=0)                       # [G-1, D]
            d_other = torch.cdist(allp, O).min(dim=1).values     # nearest other-grade centroid
            margin = d_other - d_own
            weights[g] = torch.softmax(margin / (margin.std().clamp(min=1e-6) * TAU), dim=0)
        self._bank, self._w = bank, weights

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
        dst = torch.randperm(n)[:k]
        src = torch.multinomial(w, k, replacement=True)
        out = features.clone()
        out[dst] = bank[src].to(features.device, features.dtype)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
