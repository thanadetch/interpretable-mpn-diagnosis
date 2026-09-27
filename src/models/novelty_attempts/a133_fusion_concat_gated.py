"""a133 — NAIVE backbone-fusion baseline: concat all 3 frozen FMs, then the exact
ABMIL head. This is the fusion CONTROL that smarter fusion designs must beat,
and it tests the core question: does adding UNI2-h + TITAN evidence to the Virchow2
gated-MIL help the grade at all (vs the single-backbone baseline)?

Input: data_fused bags = [N, 3584] = [virchow2 0:1280 | uni2 1280:2816 | titan 2816:3584]
(built by scripts/build_fused_features.py; patch-order verified 1330/1330).

IMPORTANT (trainer interface): the trainer forces input_dim=1280 (virchow2 cfg) for
every novelty (train_grading_reti.py:1015-1019), so we IGNORE the passed input_dim and
build the first Linear at the hardcoded fused width FUSED_DIM=3584 in __init__ — this
keeps all params present before the optimizer is constructed (no lazy-init pitfall).

Architecture = baseline ABMIL with a 3584->128 bottleneck:
  Linear(3584->128)+ReLU+Dropout(0.5) -> Ilse gated attn (V=tanh,U=sigmoid,W:128->1)
  -> softmax over patches -> attn-weighted mean -> Linear(128->1). RAW logit.

Ablation companion a134 = same head but Virchow2-only slice (= reproduces the baseline
on data_fused, isolating the fusion as the only difference).
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSED_DIM = 3584
OFFSETS = {"virchow2": (0, 1280), "uni2": (1280, 2816), "titan": (2816, 3584)}


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,          # forced to 1280 by trainer; intentionally ignored
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        backbones: Tuple[str, ...] = ("virchow2", "uni2", "titan"),
    ) -> None:
        super().__init__()
        assert num_classes == 1
        self.backbones = tuple(backbones)
        self.in_dim = sum(OFFSETS[b][1] - OFFSETS[b][0] for b in self.backbones)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _select(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] <= OFFSETS["virchow2"][1] or self.backbones == ("virchow2",):
            return features[:, OFFSETS["virchow2"][0]:OFFSETS["virchow2"][1]]
        parts = [features[:, OFFSETS[b][0]:OFFSETS[b][1]] for b in self.backbones]
        return torch.cat(parts, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = self._select(features)
        h = self.bottleneck(x)
        gated = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        attn = F.softmax(gated, dim=0)
        agg = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        out = self.classifier(agg)
        if return_attention:
            return out, attn, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
              backbones=("virchow2", "uni2", "titan"))
