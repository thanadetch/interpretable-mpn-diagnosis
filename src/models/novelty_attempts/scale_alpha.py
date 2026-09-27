# RETIRED — DO NOT RUN. The alpha lookup below (2.00 / 1.75 / 1.00 / 1.50) was chosen by
# sweeping alpha over the CV OUT-OF-FOLD TEST predictions of all 1,330 ROIs, so the table has
# seen every label it would later be scored on — under CV and under the single split alike,
# because the 259 test ROIs are a subset of those 1,330. All 36 runs were deleted 2026-08-19.
# The leak-free replacement is scripts/scale_alpha_clean.py, which refits the rule on each
# fold's VALIDATION split and applies it to that fold's test split.

"""scale_alpha — set the entmax order from the ROI's MEASURED physical scale (no learning).

THE RULE
--------
`src/tools/measure_scalebar_px.py` recovers um/px for each ROI by measuring the scale bar in
pixels. Sweeping the pooling operator at inference over the trained ASGAP checkpoints
(`scripts/alpha_vs_scale.py`) gave the alpha with the lowest error in each scale band:

    um/px  < 0.45   (patch covers ~60-100 um, zoomed in)   -> alpha 2.00   on all 3 backbones
    0.45 <= < 0.70  (~100-160 um)                          -> alpha 1.75
    um/px >= 0.70   (~160-600 um, zoomed out)              -> alpha 1.00   on all 3 backbones
    unmeasurable (46% of ROIs; dark stain hides the bar)   -> alpha 1.50   (the ASGAP default)

Nothing is learned: alpha is a lookup, so this adds ZERO parameters over ASGAP.

THE CONFOUND THIS MUST SURVIVE — read before believing any gain
---------------------------------------------------------------
The scale bands are almost disjoint in grade: the zoomed-in band holds 564 G0 and 300 G3 ROIs but
only 9 G1, while the zoomed-out band is 192 G1 + 291 G2. Controlling for grade, the best alpha
does NOT move with scale (G0 prefers 2.00 in BOTH bands, G1 prefers 1.00 in BOTH, on all three
backbones). The visible scale->alpha trend is therefore explained by grade composition: sparse
pooling pushes predictions toward the extremes, which suits the extreme grades and hurts the
middle ones. Since the pathologist zooms in further on fibrotic marrow, scale is an indirect
grade cue in THIS cohort and need not transfer to another one.

So `shuffle=True` is not optional. It applies the identical lookup table to a randomly permuted
assignment of scales, keeping the marginal distribution of alphas and destroying the
correspondence. If the shuffled control matches the real one, the scale is doing nothing and the
gain is the alpha marginal (or the confound), not the measurement.

Bag identity is recovered by a value fingerprint of the feature tensor, the same device used by
`field_mil.py`, so the trainer needs no modification. A bag whose fingerprint is not in the table
falls back to alpha 1.5 and is counted in `misses`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect

ROOT = Path(__file__).resolve().parents[3]
SCALE_CSV = ROOT / "results/scale_um_per_px.csv"
DEFAULT_ALPHA = 1.5
BANDS = ((0.45, 2.00), (0.70, 1.75))      # upper bound (exclusive) -> alpha; else 1.00
LOW_ALPHA = 1.00
VALID = (0.2, 3.0)                        # outside this range the bar detection failed


def alpha_for(um_per_px: Optional[float]) -> float:
    if um_per_px is None or not (VALID[0] <= um_per_px <= VALID[1]):
        return DEFAULT_ALPHA
    for hi, a in BANDS:
        if um_per_px < hi:
            return a
    return LOW_ALPHA


def _fingerprint(feats: torch.Tensor) -> tuple:
    f = feats.detach().to(torch.float32)
    return (int(feats.shape[0]), int(feats.shape[1]),
            float(f[0, 0]), float(f[0, -1]), float(f[-1, 0]), float(f[-1, -1]))


class _ScaleBank:
    """fingerprint(patch bag) -> alpha, built once per process for every backbone on disk."""

    _cache: dict = {}

    def __init__(self, shuffle: bool, seed: int = 0):
        self.shuffle, self.seed = bool(shuffle), int(seed)
        self.misses = 0
        self.hits = 0
        key = (shuffle, seed)
        if key not in _ScaleBank._cache:
            _ScaleBank._cache[key] = self._build()
        self.table = _ScaleBank._cache[key]

    def _build(self) -> dict:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from data.bag_dataset import GradingBagDatasetFull
        from train_grading_reti import BACKBONE_CONFIG

        um = {}
        if SCALE_CSV.exists():
            with open(SCALE_CSV) as fh:
                for r in csv.DictReader(fh):
                    try:
                        v = float(r["um_per_px"])
                    except (TypeError, ValueError):
                        continue
                    um[(r["patient"], r["stem"])] = v

        table: dict = {}
        for bb, cfg in BACKBONE_CONFIG.items():
            d = ROOT / "data" / cfg["feature_dir"]
            if not d.exists():
                continue
            try:
                ds = GradingBagDatasetFull(d)
            except Exception:
                continue
            vals, keys = [], []
            for i in range(len(ds.samples)):
                p = ds.get_slide_path(i)
                f = ds[i][0]
                if f.dim() == 3:                      # some loaders yield [1, N, D]
                    f = f.squeeze(0)
                if f.dim() != 2:
                    continue
                keys.append(_fingerprint(f))
                vals.append(um.get((p.parent.name, p.stem)))
            if self.shuffle:
                # keep the marginal distribution of scales, destroy the correspondence
                rng = np.random.default_rng(self.seed)
                vals = [vals[j] for j in rng.permutation(len(vals))]
            table.update({k: alpha_for(v) for k, v in zip(keys, vals)})
        return table

    def alpha(self, feats: torch.Tensor) -> float:
        a = self.table.get(_fingerprint(feats))
        if a is None:
            self.misses += 1
            return DEFAULT_ALPHA
        self.hits += 1
        return a


class Model(nn.Module):
    """ASGAP with the entmax order looked up from the ROI's measured scale. Zero extra params."""

    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, shuffle: bool = False, seed: int = 0):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleBank(shuffle=shuffle, seed=seed)

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        alpha = self.bank.alpha(features)
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0) if alpha <= 1.0 + 1e-6 else entmax_bisect(e, alpha)
        y = self.classifier(torch.mv(h.t(), a)).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, shuffle=False)
