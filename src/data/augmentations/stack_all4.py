"""stack_all4 - the four near-miss mechanisms of 2026-07-25/26 stacked into one recipe (train-only).

Each of these closed a different part of the gap to the champion while failing a different gate; none of
them has ever been combined. This is the full stack:

    1. DONOR RULE     donors restricted to the recipient ROI's magnification group (cutmix_scale_noise;
                      gave the val PLATEAU ~0.85 on virchow2/ASGAP instead of the champion's val spike)
    2. DESTINATION    replace the patches that look most like the ADJACENT grade, cos(x,c_adj)-cos(x,c_own)
                      (cutmix_adjclean_noise; got within 0.001 of the champion test on virchow2/ABMIL)
    3. GLOBAL VIEW    GV_FRAC of the bag becomes whole-ROI tokens of same-grade train ROIs
                      (cutmix_gvfrac; the only champion-level TEST plateau, 0.9685-0.9688 on v2/ABMIL)
    4. SCHEDULE       everything above annealed 1 -> 0 by ANNEAL_EPOCHS so training ends on real bags
                      (cutmix_anneal_noise; the winning schedule direction, and the only lever that lifted
                      titan/ASGAP test over its no-aug baseline)
    + feature_noise @ 0.05

The four act on genuinely independent parts of the recipe (which donors, which slots, what extra
information, when), so unlike the earlier two-way compositions this is not re-testing one axis twice.
`strength` = initial cutmix fraction. Metadata: `results/scalebar_results.csv` +
`data/features_<bb>_reti_no_patch/`; recipient identified by a feature-value fingerprint; donors from the
train pool only. No new data, no trainer edits. MPS-safe, deterministic given the global seed.
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch

from . import BaseAugmentation

KWARGS = dict(strength=0.8)

ANNEAL_EPOCHS = 30
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
        self._cent: Dict[int, torch.Tensor] = {}
        self._fp2scale: Dict[Tuple, str] = {}
        self._n = 1
        self._calls = 0
        self._tried = False

    def _ensure(self, pool) -> None:
        if self._tried:
            return
        self._tried = True
        self._n = max(1, len(pool))
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
        for g, bank in self._bank.items():
            c = bank.mean(dim=0)
            self._cent[g] = c / c.norm().clamp(min=1e-6)

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
        epoch = self._calls // self._n
        self._calls += 1
        decay = max(0.0, 1.0 - epoch / float(ANNEAL_EPOCHS))
        g = int(round(float(label)))
        n = features.shape[0]
        out = features
        # --- 1+2: scale-matched donors into the adjacent-grade-looking slots, annealed ---
        s_t = self.s * decay
        bank = self._donor_bank(features, g)
        if s_t > 0.0 and bank is not None and bank.shape[0] >= 1 and g in self._cent:
            k = min(max(1, int(round(s_t * n))), n)
            fn = features / features.norm(dim=1, keepdim=True).clamp(min=1e-6)
            own = fn @ self._cent[g].to(features.device, features.dtype)
            adj = None
            for gp in (g - 1, g + 1):
                if gp in self._cent:
                    sim = fn @ self._cent[gp].to(features.device, features.dtype)
                    adj = sim if adj is None else torch.maximum(adj, sim)
            dst = torch.topk(adj - own, k).indices if adj is not None else torch.randperm(n)[:k]
            src = torch.randint(bank.shape[0], (k,))
            out = features.clone()
            out[dst] = bank[src].to(features.device, features.dtype)
        # --- 3: global-view tokens, annealed ---
        gv_t = GV_FRAC * decay
        gbank = self._gbank.get(g) if self._gbank else None
        if gv_t > 0.0 and gbank is not None and gbank.shape[0] >= 1 and gbank.shape[1] == features.shape[1]:
            kg = min(max(1, int(round(gv_t * n))), n)
            dstg = torch.randperm(n)[:kg]
            srcg = torch.randint(gbank.shape[0], (kg,))
            if out is features:
                out = features.clone()
            out[dstg] = gbank[srcg].to(features.device, features.dtype)
        if out.shape[0] >= 2:
            std = out.std(dim=0, keepdim=True)
            out = out + torch.randn_like(out) * (SIGMA * std)
        return out, float(label)
