"""subspace_dropout - dropout applied ONLY to the nuisance subspace (train-only, signal-preserving).

Same signal-preserving principle as subspace_noise (which helps slide-level TITAN), but a different
regulariser: instead of ADDING Gaussian noise in the nuisance complement, we DROP OUT nuisance
dimensions (standard dropout on the nuisance component only), leaving the grade-signal subspace
span{c_g - mean} untouched. This forces the model not to rely on any particular nuisance direction
while fully preserving the grade signal.

  h = proj_signal(h) + h_perp ;  h' = proj_signal(h) + dropout(h_perp, p)

`strength` = dropout rate p on the nuisance component. Label preserved. MPS-safe, deterministic.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.2)


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.2, max_bank: int = 20000):
        self.p = float(strength)
        self.max_bank = int(max_bank)
        self._U: Optional[torch.Tensor] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        chunks = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            chunks[g].append(item[0].float())
        cents = []
        for g, lst in chunks.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cents.append(allp.mean(dim=0))
        if len(cents) < 2:
            return
        C = torch.stack(cents, dim=0)
        M = C - C.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(M, full_matrices=False)
        rank = int((S > 1e-6 * S.max()).sum().item())
        self._U = Vh[:rank].contiguous()

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.p <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._U is None:
            return features, float(label)
        U = self._U.to(features.device, features.dtype)          # [r, D]
        sig = (features @ U.t()) @ U                             # signal component
        perp = features - sig                                    # nuisance component
        # inverted dropout on the nuisance component
        mask = (torch.rand_like(perp) >= self.p).to(features.dtype) / (1.0 - self.p)
        pert = perp * mask
        pert = pert - (pert @ U.t()) @ U                         # re-remove signal leakage from the mask
        return sig + pert, float(label)
