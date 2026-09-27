"""patient_pc_shift - bag-coherent shift along the EMPIRICAL patient-effect directions (train-only).

Follow-up to the 2026-07-25 negative ``bag_nuisance_shift``, which shifted each bag by a RANDOM vector in
the nuisance subspace and was inert. In ~1500 dimensions a random nuisance direction is almost orthogonal
to the handful of directions along which patients actually differ, so that test perturbed the model
off-manifold - it could simply ignore it. This module makes the same perturbation ON-manifold: it shifts
along the directions the training patients really vary in.

  1. per-patient mean feature m_p over its training patches; centre them -> P [n_patients, D]
     (this is the empirical patient / stain-batch / scanner effect);
  2. project OUT the grade-signal subspace span{c_g - mean_g} so the shift cannot move the grade;
  3. top-k principal directions W of the residual, with the observed per-component std sigma_j;
  4. per bag: v = strength * sum_j z_j sigma_j W_j , z ~ N(0,1) - ONE vector added to EVERY patch, so it
     survives attention pooling (unlike i.i.d. per-patch noise).

strength = 1.0 is a full patient-sized displacement; default 0.5. If an on-manifold, pooling-surviving,
signal-preserving domain shift is ALSO inert, the nuisance-perturbation family is closed for good.
MPS-safe, deterministic given the global seed, no trainer edits.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.5)

NPC = 8


def _patient_of(pool, i: int) -> Optional[str]:
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        pt_path = ds.samples[idxs[i]][0] if (ds is not None and idxs is not None) else pool.samples[i][0]
        return pt_path.parent.name
    except Exception:
        return None


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.5, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._W: Optional[torch.Tensor] = None       # [k, D] patient-effect directions
        self._sig: Optional[torch.Tensor] = None     # [k] observed std along each
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        by_pat: Dict[str, list] = defaultdict(list)
        by_grade: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            by_pat[_patient_of(pool, i) or f"_idx{i}"].append(feats.mean(dim=0))
            by_grade[int(round(float(item[1])))].append(feats)
        if len(by_pat) < 3 or len(by_grade) < 2:
            return
        P = torch.stack([torch.stack(v, dim=0).mean(dim=0) for v in by_pat.values()], dim=0)  # [n_p, D]
        P = P - P.mean(dim=0, keepdim=True)

        cents = []
        for g, lst in by_grade.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            cents.append(allp.mean(dim=0))
        C = torch.stack(cents, dim=0)
        Md = C - C.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(Md, full_matrices=False)
        rank = int((S > 1e-6 * S.max()).sum().item())
        U = Vh[:rank]                                  # [r, D] grade-signal basis
        P = P - (P @ U.t()) @ U                        # patient effects with the grade signal removed

        _, Sp, Vp = torch.linalg.svd(P, full_matrices=False)
        k = min(NPC, int((Sp > 1e-6 * Sp.max()).sum().item()))
        if k < 1:
            return
        self._W = Vp[:k].contiguous()                                  # [k, D]
        self._sig = (Sp[:k] / max(1.0, (P.shape[0] - 1) ** 0.5)).contiguous()   # [k]

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._W is None or self._sig is None:
            return features, float(label)
        W = self._W.to(features.device, features.dtype)     # [k, D]
        sig = self._sig.to(features.device, features.dtype)  # [k]
        z = torch.randn(W.shape[0], device=features.device, dtype=features.dtype)
        v = ((z * sig * self.s).unsqueeze(0) @ W)            # [1, D]
        return features + v, float(label)
