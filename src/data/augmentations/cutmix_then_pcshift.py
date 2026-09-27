"""cutmix_then_pcshift - champion CutMix + an ON-MANIFOLD patient-effect shift (train-only).

Stage 1 = the champion's uniform within-grade CutMix @ ``strength``. Stage 2 = ``patient_pc_shift``'s
bag-coherent displacement along the EMPIRICAL patient-effect directions (grade-signal removed), instead of
the champion's i.i.d. Gaussian noise or the random-direction offset of cutmix_then_bagshift (both of which
turned out interchangeable / inert on 2026-07-25).

This is the last stage-2 candidate that is qualitatively different: a perturbation that is (a) bag-coherent
so it survives pooling, (b) on the manifold the data actually varies along, and (c) orthogonal to the grade
signal. If it too lands at champion-minus-a-little, the "CutMix carries everything, stage 2 is
interchangeable" statement is complete and the whole perturbation family can be closed in the thesis.

`strength` = cutmix fraction (champion default 0.8); SHIFT below = patient-shift magnitude.
MPS-safe, deterministic given the global seed, no trainer edits.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SHIFT = 0.5
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

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._W: Optional[torch.Tensor] = None
        self._sig: Optional[torch.Tensor] = None
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
        bank, cents = {}, []
        for g, lst in by_grade.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
            cents.append(allp.mean(dim=0))
        self._bank = bank
        if len(by_pat) < 3 or len(cents) < 2:
            return
        C = torch.stack(cents, dim=0)
        Md = C - C.mean(dim=0, keepdim=True)
        _, S, Vh = torch.linalg.svd(Md, full_matrices=False)
        rank = int((S > 1e-6 * S.max()).sum().item())
        U = Vh[:rank]
        P = torch.stack([torch.stack(v, dim=0).mean(dim=0) for v in by_pat.values()], dim=0)
        P = P - P.mean(dim=0, keepdim=True)
        P = P - (P @ U.t()) @ U
        _, Sp, Vp = torch.linalg.svd(P, full_matrices=False)
        k = min(NPC, int((Sp > 1e-6 * Sp.max()).sum().item()))
        if k < 1:
            return
        self._W = Vp[:k].contiguous()
        self._sig = (Sp[:k] / max(1.0, (P.shape[0] - 1) ** 0.5)).contiguous()

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        out = features
        if self.s > 0.0 and self._bank is not None:
            bank = self._bank.get(int(round(float(label))))
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if self._W is not None and self._sig is not None:
            W = self._W.to(out.device, out.dtype)
            sig = self._sig.to(out.device, out.dtype)
            z = torch.randn(W.shape[0], device=out.device, dtype=out.dtype)
            out = out + ((z * sig * SHIFT).unsqueeze(0) @ W)
        return out, float(label)
