"""cutmix_scale_noise - within-grade CutMix restricted to SAME-MAGNIFICATION donors + noise (train-only).

Every donor axis explored so far was FEATURE-GEOMETRIC (prototype / margin / NN / FPS / 2-donor) or
sample-count based (patient-balanced), and uniform-over-patches beat all of them
([[asgap-wants-diverse-donors]]). This module opens the one axis that is neither: DOMAIN METADATA.

This cohort has genuine acquisition heterogeneity - the ROIs were captured at different magnifications
(scalebar 20 / 50 / 100 / 200 um, plus images where no scalebar was detected). Uniform within-grade
CutMix therefore transplants patches ACROSS magnifications, producing bags that are physically
impossible (a single microscope field cannot contain 20um-scale and 200um-scale tissue). Hypothesis:
physically-coherent recombination (donors from the same magnification group) is a better regulariser
than diversity-maximal but physically invalid recombination.

    1. within-grade CutMix @ ``strength``, donors drawn uniformly from the patches of the SAME
       (grade, scalebar-group) as the recipient ROI  (fallback: the full within-grade bank when the
       matched bank is too small)
    2. feature_noise @ 0.05  (the champion's isotropic bag-std jitter)

Everything else is byte-identical to the champion ``cutmix_then_noise`` - the ONLY change is the donor
restriction, so the comparison isolates the magnification axis. The opposite direction (donors forced
from a DIFFERENT magnification = scale-invariance pressure) is ``cutmix_crossscale_noise``.

Magnification is read from ``results/scalebar_results.csv`` (patient, filename -> scalebar_micron;
1329/1330 ROIs covered) and every grade contains several magnification groups, so the restriction is
informative rather than a relabelling of the grade. The recipient ROI is identified drop-in by a
value fingerprint of its feature matrix (the trainer passes only ``features``), so no trainer/dataset
edits are needed. `strength` = cutmix replacement fraction. MPS-safe, deterministic given the seed.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SIGMA = 0.05
MIN_BANK = 200                     # matched bank smaller than this -> fall back to the full grade bank
SCALE_CSV = Path(__file__).resolve().parents[3] / "results" / "scalebar_results.csv"


def _scale_table() -> Dict[Tuple[str, str], str]:
    """(patient_dir_name, roi_stem) -> scalebar group ('20'/'50'/'100'/'200'/'unknown')."""
    table: Dict[Tuple[str, str], str] = {}
    try:
        with open(SCALE_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                table[(row["patient"], Path(row["filename"]).stem)] = row["scalebar_micron"]
    except Exception:
        pass
    return table


def _path_of(pool, i: int) -> Optional[Path]:
    """Recover the .pt path for pool item i, drop-in via the train Subset."""
    try:
        ds = getattr(pool, "dataset", None)
        idxs = getattr(pool, "indices", None)
        if ds is not None and idxs is not None:
            return ds.samples[idxs[i]][0]
        return pool.samples[i][0]
    except Exception:
        return None


def _fp(features: torch.Tensor) -> Tuple:
    """Value fingerprint of a bag (exact element values, no accumulation -> device-independent)."""
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
        self._bank: Optional[Dict[int, torch.Tensor]] = None                    # grade -> [P, D]
        self._sbank: Optional[Dict[Tuple[int, str], torch.Tensor]] = None       # (grade, scale) -> [P, D]
        self._fp2scale: Dict[Tuple, str] = {}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        table = _scale_table()
        by_g: Dict[int, list] = defaultdict(list)
        by_gs: Dict[Tuple[int, str], list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            path = _path_of(pool, i)
            scale = table.get((path.parent.name, path.stem), "unknown") if path is not None else "unknown"
            self._fp2scale[_fp(feats)] = scale
            by_g[g].append(feats)
            by_gs[(g, scale)].append(feats)

        def _cat(lst, cap):
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > cap:
                allp = allp[torch.randperm(allp.shape[0])[:cap]]
            return allp

        self._bank = {g: _cat(lst, self.max_bank) for g, lst in by_g.items()}
        self._sbank = {k: _cat(lst, self.max_bank) for k, lst in by_gs.items()}

    def _donor_bank(self, features: torch.Tensor, g: int) -> Optional[torch.Tensor]:
        scale = self._fp2scale.get(_fp(features))
        if scale is not None and self._sbank is not None:
            bank = self._sbank.get((g, scale))
            if bank is not None and bank.shape[0] >= MIN_BANK:
                return bank
        return self._bank.get(g) if self._bank is not None else None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        out = features
        if self.s > 0.0 and pool is not None and features.shape[0] >= 4:
            self._ensure(pool)
            g = int(round(float(label)))
            bank = self._donor_bank(features, g)
            if bank is not None and bank.shape[0] >= 1:
                n = features.shape[0]
                k = min(max(1, int(round(self.s * n))), n)
                dst = torch.randperm(n)[:k]
                src = torch.randint(bank.shape[0], (k,))
                out = features.clone()
                out[dst] = bank[src].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
