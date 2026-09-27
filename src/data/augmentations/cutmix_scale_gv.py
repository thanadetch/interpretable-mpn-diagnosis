"""cutmix_scale_gv - scale-matched CutMix (val plateau) + global-view tokens (test plateau) (train-only).

The champion cutmix_then_noise was itself found by COMPOSING the two mechanisms that each won a different
gate. The 2026-07-25 rounds produced exactly that situation again, with two mechanisms whose strengths are
complementary and whose weaknesses are opposite:

  * ``cutmix_scale_noise`` (donors restricted to the recipient ROI's magnification): a flat val PLATEAU on
    virchow2/ASGAP (0.841-0.852 across strengths 0.70-0.90) but test ~0.955-0.961, a touch under champion;
  * ``cutmix_gvfrac`` (a small dose of whole-ROI / low-magnification tokens on top of CutMix): a flat TEST
    plateau at champion level on virchow2/ABMIL (0.9685/0.9687/0.9688 at doses 0.05/0.10/0.15) but val
    0.80-0.83, clearly under champion.

So: take the scale-matched donor rule for stage 1 and add the global-view dose as stage 2.

    1. within-grade CutMix @ ``strength``, donors from the SAME scalebar group as the recipient ROI
    2. replace GV_FRAC of the bag with whole-ROI (no_patch) embeddings of same-grade TRAIN ROIs
    3. feature_noise @ 0.05

Note this also makes the two magnification levels explicit and consistent: the transplanted patches match
the recipient's magnification, and the only deliberately different-magnification tokens are the global
ones. `strength` = cutmix fraction; GV_FRAC = global-token dose. Metadata from
``results/scalebar_results.csv``; recipient identified by a feature-value fingerprint; donors from the
train pool only. No new data, no trainer edits. MPS-safe, deterministic.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

GV_FRAC = 0.10
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
        return ds.samples[idxs[i]][0] if (ds is not None and idxs is not None) else pool.samples[i][0]
    except Exception:
        return None


def _no_patch_path(p: Path) -> Optional[Path]:
    parts = list(p.parts)
    for j, part in enumerate(parts):
        if part.startswith("features_") and not part.endswith("_no_patch"):
            parts[j] = part + "_no_patch"
            return Path(*parts)
    return None


def _fp(features: torch.Tensor) -> Tuple:
    n, d = features.shape
    return (int(n), int(d), float(features[0, 0]), float(features[0, -1]),
            float(features[-1, 0]), float(features[-1, -1]))


class Augmentation(BaseAugmentation):
    requires_regression = False

    def __init__(self, strength: float = 0.8, max_bank: int = 20000):
        self.s = float(strength)
        self.max_bank = int(max_bank)
        self._bank: Optional[Dict[int, torch.Tensor]] = None
        self._sbank: Optional[Dict[Tuple[int, str], torch.Tensor]] = None
        self._gbank: Optional[Dict[int, torch.Tensor]] = None
        self._fp2scale: Dict[Tuple, str] = {}
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        table = _scale_table()
        by_g: Dict[int, list] = defaultdict(list)
        by_gs: Dict[Tuple[int, str], list] = defaultdict(list)
        gv: Dict[int, list] = defaultdict(list)
        for i in range(len(pool)):
            item = pool[i]
            feats = item[0].float()
            g = int(round(float(item[1])))
            path = _path_of(pool, i)
            scale = table.get((path.parent.name, path.stem), "unknown") if path is not None else "unknown"
            self._fp2scale[_fp(feats)] = scale
            by_g[g].append(feats)
            by_gs[(g, scale)].append(feats)
            gp = _no_patch_path(path) if path is not None else None
            if gp is not None and gp.exists():
                try:
                    data = torch.load(gp, map_location="cpu", weights_only=False)
                    vec = data["feats"] if isinstance(data, dict) else data
                    gv[g].append(vec.float().reshape(1, -1))
                except Exception:
                    pass

        def _cat(lst, cap):
            allp = torch.cat(lst, dim=0)
            if allp.shape[0] > cap:
                allp = allp[torch.randperm(allp.shape[0])[:cap]]
            return allp

        self._bank = {g: _cat(l, self.max_bank) for g, l in by_g.items()}
        self._sbank = {k: _cat(l, self.max_bank) for k, l in by_gs.items()}
        self._gbank = {g: torch.cat(l, dim=0) for g, l in gv.items() if l}

    def _donor_bank(self, features: torch.Tensor, g: int) -> Optional[torch.Tensor]:
        scale = self._fp2scale.get(_fp(features))
        if scale is not None and self._sbank is not None:
            bank = self._sbank.get((g, scale))
            if bank is not None and bank.shape[0] >= MIN_BANK:
                return bank
        return self._bank.get(g) if self._bank is not None else None

    def __call__(self, features: torch.Tensor, label: float, pool=None) -> Tuple[torch.Tensor, float]:
        if pool is None or features.shape[0] < 4:
            return features, float(label)
        self._ensure(pool)
        g = int(round(float(label)))
        n = features.shape[0]
        out = features
        bank = self._donor_bank(features, g)
        if self.s > 0.0 and bank is not None and bank.shape[0] >= 1:
            k = min(max(1, int(round(self.s * n))), n)
            dst = torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        gbank = self._gbank.get(g) if self._gbank else None
        if GV_FRAC > 0.0 and gbank is not None and gbank.shape[0] >= 1 and gbank.shape[1] == features.shape[1]:
            kg = min(max(1, int(round(GV_FRAC * n))), n)
            dstg = torch.randperm(n)[:kg]
            srcg = torch.randint(gbank.shape[0], (kg,))
            if out is features:
                out = features.clone()
            out[dstg] = gbank[srcg].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
