"""ASGAP with the whole-ROI ("global") view entering BEFORE pooling, never as a weighted instance.

MOTIVATION
    Earlier field-conditioned variants put the global view either AFTER pooling (a340,
    ``readout="concat"``) or as its own token in a transformer (a381). Both share a limitation:
    when the global view arrives after pooling it cannot influence WHICH patches are selected —
    the selection has already happened — and when it arrives as a token it competes for attention
    mass like any other instance.

    This module takes the third position: the global view conditions the attention SCORER, so it
    shapes the selection, but it is never scored and never receives a weight. The question the
    attention answers changes from "does this patch matter?" to "given this field, does this patch
    matter?".

MODES
    ``mode="context"``  (default)
        g conditions the scorer only; the pooled vector is built from the patches alone.

            h = bottleneck(x)                      [N, H]
            g = bottleneck(field)                  [1, H]   <- SHARED bottleneck
            e = W( V(h+g) * U(h+g) )               scores see the field
            a = entmax(e, alpha)
            z = h^T a                              pooled from h, NOT h+g

        The field influences selection and nothing else: no term of z comes from g.

    ``mode="infuse"``
        The same conditioning, but the pooled vector is built from the conditioned features:

            z = (h+g)^T a

        Here the field does enter the representation, uniformly across patches (a common shift),
        still without being scored.

PARAMETER COUNT
    Both modes reuse the patch bottleneck for the field, so the parameter count is IDENTICAL to
    ABMIL and ASGAP (197,250 at input_dim=1280, num_classes=1). The comparison against ASGAP
    therefore differs in exactly one thing — whether the scorer sees the field — with no capacity
    confound to control for.

FIELD SOURCE
    The whole-ROI embedding is resolved from ``data/features_<backbone>_reti_no_patch`` through the
    shared value-fingerprint bank in ``field_mil`` (same mechanism as a381). ``field="mean"`` swaps
    it for the bag's own patch mean, which carries no information the bag does not already have.

    Requires the ``_no_patch`` features to have been extracted for the backbone in use.

Self-contained, permutation/size-invariant, deterministic at eval, MPS-safe, no new deps.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .a215_adaptive_sparse_gated_attention_pooling import entmax_bisect
from .field_mil import _bank, _fingerprint

MODES = ("context", "infuse")
FIELD_SOURCES = ("roi", "mean")


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, num_classes: int = 1, hidden_dim: int = 128,
                 dropout: float = 0.5, mode: str = "context", field: str = "roi",
                 data_root: Optional[str] = None):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if field not in FIELD_SOURCES:
            raise ValueError(f"field must be one of {FIELD_SOURCES}")
        self.mode, self.field = mode, field
        self.data_root = Path(data_root or os.environ.get("MPN_DATA_ROOT", "data"))

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))       # sigmoid -> 0.5 -> alpha = 1.5
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._bank_obj = None
        self._hit = 0
        self._miss = 0
        self._warned = False

    # ── field vector ─────────────────────────────────────────────────────
    def _field(self, features: torch.Tensor) -> torch.Tensor:
        if self.field == "mean":
            return features.mean(dim=0)
        if self._bank_obj is None:
            self._bank_obj = _bank(int(features.shape[1]), self.data_root)
        vec = self._bank_obj.table.get(_fingerprint(features))
        if vec is None:
            self._miss += 1
            if not self._warned:
                self._warned = True
                print("  ⚠ field_before_pool: bag missing from the field bank — using the patch "
                      "mean. Expected only when a feature-space augmentation is active.")
            return features.mean(dim=0)
        self._hit += 1
        return vec.to(features.device, features.dtype)

    def field_hit_rate(self) -> float:
        n = self._hit + self._miss
        return float(self._hit) / n if n else 0.0

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, features, return_attention: bool = False, metrics=None
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if features.dim() == 3:
            features = features.squeeze(0)
        g_raw = self._field(features)

        h = self.bottleneck(features)                          # [N, H]
        g = self.bottleneck(g_raw.unsqueeze(0))                # [1, H]  shared weights
        c = h + g                                              # field-conditioned view

        e = self.attention_W(self.attention_V(c) * self.attention_U(c)).squeeze(-1)
        alpha = 1.0 + torch.sigmoid(self.alpha_raw)            # in (1, 2)
        a = entmax_bisect(e, alpha)

        pooled_from = h if self.mode == "context" else c
        z = torch.mv(pooled_from.t(), a)
        y = self.classifier(z).view(-1)

        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, mode="context", field="roi")
