"""scale_gated_alpha — learn, from the ROI's measured um/px, how to mix five entmax orders.

THE IDEA
--------
Instead of hand-setting alpha per scale band (a407), pool the SAME attention logits at five fixed
orders alpha = 1.00 / 1.25 / 1.50 / 1.75 / 2.00 and let a gate decide how much of each to use,
conditioned only on the ROI's measured physical scale:

    e            = gated-attention logits                       [N]
    z_k          = sum_i entmax_{alpha_k}(e)_i h_i              five pooled descriptors
    w            = softmax( W [ s , u ] + b )                    five mixture weights
    z            = sum_k w_k z_k
    y            = Linear(z)

with  s = (log10(um/px) - LOG_REF) / LOG_SCALE   standardised scale, 0 when unmeasurable
      u = 1 when the scale bar could not be read (46% of ROIs), else 0

The gate is `Linear(2 -> 5)` = **15 parameters**. It is the smallest object that can express
"which sparsity level suits this magnification", and because `u` is a separate input it can learn
a different mixture for the unmeasurable ROIs rather than being forced onto a fallback.

WHAT THE EVIDENCE ALREADY SAYS — read before hoping
---------------------------------------------------
The leak-free hand-set version (a407, alpha fitted on each fold's val) gained 0 / +14 / +16 ROI
of 1330 under CV and -5 / +3 / -2 on the single split; the alpha that val selects is unstable
across folds and contradictory across backbones (virchow2 picks 1.00 for the zoomed-in band where
titan picks 2.00). Controlling for grade, the best alpha does not move with scale at all: G0
prefers 2.00 and G1 prefers 1.00 in BOTH scale bands, on all three encoders. So the prior is that
this gate has little to learn.

Its value is diagnostic. Unlike a fixed rule, a learned gate REPORTS what it found: if the
scale coefficients collapse toward zero and the mixture becomes constant, the model itself is
saying the measurement is uninformative, which is a clean and interpretable negative.

THE CONTROL
-----------
`condition=False` feeds the gate a constant input. It can then learn one GLOBAL mixture of the
five orders but cannot condition on anything. That isolates the actual question — does knowing
the magnification help, beyond simply blending several sparsity levels? — because blending alone
is already known to shift predictions toward the middle grades and lift G1.

`gate_report()` returns the learned weights at a few scales so the mapping can be inspected
directly instead of inferred from accuracy.
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
ALPHAS = (1.00, 1.25, 1.50, 1.75, 2.00)
LOG_REF, LOG_SCALE = -0.30, 0.35          # log10(um/px): ~0.5 um/px centre, ~0.2-2.7 spread
VALID = (0.2, 3.0)


def _fingerprint(feats: torch.Tensor) -> tuple:
    f = feats.detach().to(torch.float32)
    return (int(feats.shape[0]), int(feats.shape[1]),
            float(f[0, 0]), float(f[0, -1]), float(f[-1, 0]), float(f[-1, -1]))


class _ScaleFeatureBank:
    """fingerprint(bag) -> (standardised log scale, unknown flag)."""

    _cache: dict = {}

    def __init__(self):
        if "t" not in _ScaleFeatureBank._cache:
            _ScaleFeatureBank._cache["t"] = self._build()
        self.table = _ScaleFeatureBank._cache["t"]
        self.hits = self.misses = 0

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
                    if VALID[0] <= v <= VALID[1]:
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
            for i in range(len(ds.samples)):
                f = ds[i][0]
                if f.dim() == 3:
                    f = f.squeeze(0)
                if f.dim() != 2:
                    continue
                p = ds.get_slide_path(i)
                v = um.get((p.parent.name, p.stem))
                table[_fingerprint(f)] = ((0.0, 1.0) if v is None
                                          else ((np.log10(v) - LOG_REF) / LOG_SCALE, 0.0))
        return table

    def feat(self, feats: torch.Tensor) -> tuple:
        v = self.table.get(_fingerprint(feats))
        if v is None:
            self.misses += 1
            return (0.0, 1.0)
        self.hits += 1
        return v


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, condition: bool = True):
        super().__init__()
        self.condition = bool(condition)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(2, len(ALPHAS))      # 15 params
        nn.init.zeros_(self.gate.weight)           # start as a uniform mixture of the five orders
        nn.init.zeros_(self.gate.bias)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.bank = _ScaleFeatureBank()

    def _weights(self, s: float, u: float, device, dtype) -> torch.Tensor:
        x = torch.tensor([s, u] if self.condition else [0.0, 0.0], device=device, dtype=dtype)
        return torch.softmax(self.gate(x), dim=0)

    @torch.no_grad()
    def gate_report(self, um_values=(0.27, 0.36, 0.54, 1.06, 2.70, None)) -> dict:
        """The learned mixture at a few magnifications — inspect the mapping directly."""
        out = {}
        for v in um_values:
            s, u = ((0.0, 1.0) if v is None else ((np.log10(v) - LOG_REF) / LOG_SCALE, 0.0))
            w = self._weights(s, u, self.gate.weight.device, self.gate.weight.dtype)
            out["unknown" if v is None else f"{v:.2f}"] = [round(float(x), 3) for x in w]
        return out

    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        s, u = self.bank.feat(features)
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        w = self._weights(s, u, h.device, h.dtype)

        z = 0.0
        amix = 0.0
        for k, al in enumerate(ALPHAS):
            a = F.softmax(e, dim=0) if al <= 1.0 + 1e-6 else entmax_bisect(e, al)
            z = z + w[k] * torch.mv(h.t(), a)
            amix = amix + w[k] * a
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, amix, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, condition=True)
