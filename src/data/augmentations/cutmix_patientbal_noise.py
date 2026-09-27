"""cutmix_patientbal_noise - within-grade CutMix with PATIENT-BALANCED donor sampling + noise (train-only).

The champion cutmix_then_noise samples donors uniformly over PATCHES within a grade, so patients that
contribute more ROIs/patches dominate the donor pool. This variant samples donors uniformly over PATIENTS
first (each patient in the grade equally likely), then a random patch from that patient — maximizing
inter-patient donor diversity, the logical extreme of the ASGAP-prefers-diverse-donors finding. Then the
champion's isotropic all-patch Gaussian noise (bag-std * SIGMA). `strength` = cutmix fraction.

Patient identity is recovered drop-in from the train Subset (pool.dataset.samples[pool.indices[i]][0].parent),
so no trainer/dataset edits are needed. MPS-safe, deterministic given seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)
SIGMA = 0.05


def _patient_of(pool, i: int) -> Optional[str]:
    """Recover the patient id (parent dir name) for pool item i, drop-in via the Subset."""
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        if ds is not None and idxs is not None:
            pt_path = ds.samples[idxs[i]][0]
        else:  # pool is the full dataset itself
            pt_path = pool.samples[i][0]
        return pt_path.parent.name
    except Exception:
        return None


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_per_patient: int = 4000):
        self.s = float(strength)
        self.max_per_patient = int(max_per_patient)
        # bank[g] = list of per-patient patch tensors; each entry is [n_p, D]
        self._bank: Optional[Dict[int, List[torch.Tensor]]] = None
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        # group patches by (grade, patient)
        by_gp: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for i in range(len(pool)):
            item = pool[i]
            g = int(round(float(item[1])))
            pat = _patient_of(pool, i) or f"_idx{i}"  # fall back to per-bag if patient unknown
            by_gp[g][pat].append(item[0].float())
        bank: Dict[int, List[torch.Tensor]] = {}
        for g, pats in by_gp.items():
            plist = []
            for pat, lst in pats.items():
                allp = torch.cat(lst, dim=0)
                if allp.shape[0] > self.max_per_patient:
                    allp = allp[torch.randperm(allp.shape[0])[: self.max_per_patient]]
                plist.append(allp)
            bank[g] = plist
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if self.s <= 0.0 or pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        if self._bank is None:
            return features, float(label)
        g = int(round(float(label)))
        plist = self._bank.get(g)
        if not plist:
            return features, float(label)
        n = features.shape[0]
        k = min(max(1, int(round(self.s * n))), n)
        np_ = len(plist)
        # patient-balanced: pick a random patient per donor, then a random patch from that patient
        pat_ids = torch.randint(np_, (k,))
        donors = torch.empty((k, features.shape[1]), dtype=features.dtype, device=features.device)
        for j in range(k):
            pt = plist[pat_ids[j].item()]
            r = torch.randint(pt.shape[0], (1,)).item()
            donors[j] = pt[r].to(features.device, features.dtype)
        dst = torch.randperm(n)[:k]
        out = features.clone()
        out[dst] = donors
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        out = out + torch.randn_like(out) * (std * SIGMA)
        return out, float(label)
