"""subspace_noise_lda - nuisance-subspace feature noise, signal subspace = Fisher/LDA directions.

Same idea as subspace_noise (add noise only ORTHOGONAL to the grade-signal subspace), but with a better
signal subspace. subspace_noise used span{centroid_g - mean} (between-class scatter directions only,
ignoring within-class spread). This uses regularised Fisher LDA — the top eigenvectors of S_w^{-1} S_b —
which are the MAXIMALLY grade-discriminative directions (they account for within-class scatter). Protecting
those, and injecting noise only in their orthogonal complement, should preserve the grade signal more
precisely and perturb only true nuisance variation.

  S_b = between-class scatter, S_w = within-class scatter (shrinkage-regularised), signal basis =
  orthonormalised top-(#grades-1) LDA directions.  noise ~ N(0,(within-bag std * strength)^2), projected
  onto the complement of that basis.  Label preserved.

`strength` = noise magnitude (fraction of per-dim std). MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.1)

SHRINK = 1e-2  # S_w shrinkage: S_w += SHRINK * mean(diag(S_w)) * I  (stabilise the inverse)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.1, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._U: Optional[torch.Tensor] = None   # [r, D] orthonormal LDA signal basis
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
        if len(chunks) < 2:
            return
        means = {}
        counts = {}
        Xc_list = []
        all_for_mean = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank // max(1, len(chunks)):
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank // max(1, len(chunks))]]
            c = allp.mean(dim=0, keepdim=True)     # [1, D]
            means[g] = c
            counts[g] = allp.shape[0]
            Xc_list.append(allp - c)               # centred within-class
            all_for_mean.append((c.squeeze(0), allp.shape[0]))
        D = Xc_list[0].shape[1]
        # global mean (weighted)
        tot = sum(counts.values())
        gmean = sum(means[g].squeeze(0) * counts[g] for g in means) / tot   # [D]
        # within-class scatter S_w = sum_g (X_g - c_g)^T (X_g - c_g)
        Xc = torch.cat(Xc_list, dim=0)             # [N, D]
        Sw = Xc.t() @ Xc                           # [D, D]
        Sw += SHRINK * (Sw.diagonal().mean()) * torch.eye(D)
        # between-class scatter S_b = sum_g n_g (c_g - gmean)(c_g - gmean)^T
        Sb = torch.zeros(D, D)
        for g in means:
            diff = (means[g].squeeze(0) - gmean).unsqueeze(1)   # [D,1]
            Sb += counts[g] * (diff @ diff.t())
        # top LDA directions = eigenvectors of Sw^{-1} Sb
        try:
            M = torch.linalg.solve(Sw, Sb)         # Sw^{-1} Sb  [D, D]
            evals, evecs = torch.linalg.eig(M)
            evals = evals.real; evecs = evecs.real
            r = min(len(means) - 1, D)
            idx = torch.argsort(evals, descending=True)[:r]
            W = evecs[:, idx]                      # [D, r] LDA directions (not orthonormal)
            # orthonormal basis of span(W) via QR
            Q, _ = torch.linalg.qr(W)              # [D, r]
            self._U = Q.t().contiguous()           # [r, D]
        except Exception:
            self._U = None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._U is None:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)          # [r, D]
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        noise = torch.randn_like(features) * (std * self.s)
        proj = (noise @ U.t()) @ U
        return features + (noise - proj), float(label)
