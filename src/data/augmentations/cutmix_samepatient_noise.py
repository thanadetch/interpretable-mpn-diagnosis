"""cutmix_samepatient_noise - within-PATIENT CutMix (physically real recombination) + noise (train-only).

Third point on the "physical validity of the recombination" hypothesis (with cutmix_scale_noise /
cutmix_crossscale_noise). The champion cutmix_then_noise transplants patches from ANY patient of the
same grade, so an augmented bag mixes tissue from different biopsies, different stain batches and
different microscopes - a bag that could never be acquired. This module restricts donors to OTHER ROIs
OF THE SAME PATIENT: every augmented bag is then a re-cut of one real biopsy (same tissue, stain,
scanner, magnification) - the maximally realistic pseudo-ROI, and the natural completion of the donor
axis (patient-BALANCED sampling = maximal inter-patient diversity was already tested and lost;
same-patient = the opposite extreme, never tested).

    1. CutMix @ ``strength`` with donors drawn uniformly from the recipient patient's OTHER training
       ROIs (the recipient ROI's own patches are excluded, so it is a genuine transplant; patients with
       a single training ROI fall back to the full within-grade bank)
    2. feature_noise @ 0.05  (the champion's isotropic bag-std jitter)

Prediction from [[asgap-wants-diverse-donors]]: this should help ABMIL (soft attention likes
concentrated donors) and hurt ASGAP (sparse attention needs diverse evidence) - if instead it helps
BOTH, then donor "realism" is a factor distinct from donor "concentration".

Patient identity is recovered drop-in from the train Subset; the recipient ROI is identified by a value
fingerprint of its feature matrix. No trainer edits. MPS-safe, deterministic given the seed.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SIGMA = 0.05


def _path_of(pool, i: int):
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        if ds is not None and idxs is not None:
            return ds.samples[idxs[i]][0]
        return pool.samples[i][0]
    except Exception:
        return None


def _fp(features: torch.Tensor) -> Tuple:
    n, d = features.shape
    return (
        int(n), int(d),
        float(features[0, 0]), float(features[0, -1]),
        float(features[-1, 0]), float(features[-1, -1]),
    )


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None          # grade -> [P, D] fallback bank
        self._pbank: Dict[str, torch.Tensor] = {}                     # patient -> [P, D] all its patches
        self._fp2loc: Dict[Tuple, Tuple[str, int, int]] = {}          # fp -> (patient, start, n_self)
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        by_g: Dict[int, list] = defaultdict(list)
        by_p: Dict[str, list] = defaultdict(list)
        meta: List[Tuple[Tuple, str, int]] = []                       # (fp, patient, n_patches)
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            path = _path_of(pool, i)
            pat = path.parent.name if path is not None else f"_idx{i}"
            by_g[g].append(feats)
            by_p[pat].append(feats)
            meta.append((_fp(feats), pat, feats.shape[0]))

        offset: Dict[str, int] = defaultdict(int)
        for fp, pat, n in meta:                                       # bags are concatenated in order
            self._fp2loc[fp] = (pat, offset[pat], n)
            offset[pat] += n

        self._pbank = {p: torch.cat(lst, dim=0) for p, lst in by_p.items()}
        bank: Dict[int, torch.Tensor] = {}
        for g, lst in by_g.items():
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > self.max_bank:
                allp = allp[torch.randperm(allp.shape[0])[: self.max_bank]]
            bank[g] = allp
        self._bank = bank

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            n = features.shape[0]
            k = min(max(1, int(round(self.s * n))), n)
            loc = self._fp2loc.get(_fp(features))
            donors = None
            if loc is not None:
                pat, start, n_self = loc
                pb = self._pbank.get(pat)
                if pb is not None and pb.shape[0] - n_self >= 1:      # patient has other ROIs
                    r = torch.randint(pb.shape[0] - n_self, (k,))
                    r = r + (r >= start).long() * n_self              # skip this ROI's own block
                    donors = pb[r]
            if donors is None and self._bank is not None:             # single-ROI patient -> fallback
                bank = self._bank.get(int(round(float(label))))
                if bank is not None and bank.shape[0] >= 1:
                    donors = bank[torch.randint(bank.shape[0], (k,))]
            if donors is not None:
                dst = torch.randperm(n)[:k]
                out = features.clone()
                out[dst] = donors.to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
