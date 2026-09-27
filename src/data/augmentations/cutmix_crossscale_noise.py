"""cutmix_crossscale_noise - within-grade CutMix forced ACROSS magnifications + noise (train-only).

The opposite pole of ``cutmix_scale_noise`` on the new DOMAIN-METADATA donor axis. Instead of keeping
recombination physically coherent, this one deliberately transplants patches from a DIFFERENT
magnification group (scalebar 20/50/100/200um, or the no-scalebar group) of the same grade - a
domain-randomisation pressure that forces the aggregator to grade fibrosis in a way that cannot depend
on the acquisition magnification.

    1. within-grade CutMix @ ``strength``, donors drawn uniformly from patches of the SAME grade but a
       DIFFERENT scalebar group than the recipient ROI (fallback: full within-grade bank if that grade
       has no other group with enough patches)
    2. feature_noise @ 0.05  (the champion's isotropic bag-std jitter)

Together the two modules bracket the magnification axis around the champion ``cutmix_then_noise``
(which is scale-agnostic = a mixture of both), so the pair answers whether scale-coherence, scale-
mismatch, or indifference is best. MixStyle previously attacked the same real heterogeneity through
feature STATISTICS and landed on the val<->test frontier; this attacks it through DONOR SELECTION.

`strength` = cutmix replacement fraction. Metadata from ``results/scalebar_results.csv``; recipient ROI
identified by a value fingerprint of its feature matrix. No trainer edits. MPS-safe, deterministic.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

SIGMA = 0.05
MIN_BANK = 200
SCALE_CSV = Path(__file__).resolve().parents[3] / "results" / "scalebar_results.csv"


def _scale_table() -> Dict[Tuple[str, str], str]:
    table: Dict[Tuple[str, str], str] = {}
    try:
        with open(SCALE_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                table[(row["patient"], Path(row["filename"]).stem)] = row["scalebar_micron"]
    except Exception:
        pass
    return table


def _path_of(pool, i: int) -> Optional[Path]:
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
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._sbank: Optional[Dict[Tuple[int, str], torch.Tensor]] = None
        self._scales: Dict[int, List[str]] = {}
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
        groups: Dict[int, List[str]] = defaultdict(list)
        for (g, s), bank in self._sbank.items():
            if bank.shape[0] >= MIN_BANK:
                groups[g].append(s)
        self._scales = dict(groups)

    def _donor_bank(self, features: torch.Tensor, g: int) -> Optional[torch.Tensor]:
        scale = self._fp2scale.get(_fp(features))
        others = [s for s in self._scales.get(g, []) if s != scale]
        if scale is not None and others:
            pick = others[int(torch.randint(len(others), (1,)).item())]
            bank = self._sbank.get((g, pick)) if self._sbank is not None else None
            if bank is not None and bank.shape[0] >= 1:
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
