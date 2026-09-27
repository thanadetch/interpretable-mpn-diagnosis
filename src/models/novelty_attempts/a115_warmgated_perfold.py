"""a115 — warm-started gated attention, PER-FOLD-MATCHED axis, backbone-agnostic.

The rigorous version of a113. a113 used the seed=2 axis on every run, so it is
only honest at seed=2 (the favorable fold the axis was fit to). a115 instead
loads the axis computed from THIS run's OWN fold train split: it reads `--seed`
from sys.argv and the backbone from input_dim, then loads
`prototypes_{backbone}_reti_train_seed{seed}.pt`. Each fold therefore gets its
own train-only axis -> NO leakage and a fair paired test of whether the
warm-start helps BROADLY (across folds AND backbones) or is purely a seed=2
artifact.

Pair a115 (warm, per-fold-matched) vs a114 (warm_start=False = plain) across
folds {0,1,2,3,42} on {virchow2, uni2, titan}. forward uses ONLY 'features'.
RAW logits. Deterministic at inference.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA = Path(__file__).resolve().parents[3] / "data"
_BB_BY_DIM = {1280: "virchow2", 1536: "uni2", 768: "titan"}


def _current_seed(default: int = 2) -> int:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--seed" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
        if a.startswith("--seed="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                pass
    return default


def _axis(input_dim: int) -> Optional[torch.Tensor]:
    bb = _BB_BY_DIM.get(input_dim)
    if bb is None:
        return None
    seed = _current_seed()
    path = _DATA / f"prototypes_{bb}_reti_train_seed{seed}.pt"
    if not path.exists():
        return None
    try:
        v = torch.load(path, map_location="cpu", weights_only=False)["axis"].float().view(-1)
        if v.numel() == input_dim:
            return v / v.norm().clamp(min=1e-8)
    except Exception:
        pass
    return None


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        warm_start: bool = True,
    ) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        if warm_start:
            axis = _axis(input_dim)
            if axis is not None:
                with torch.no_grad():
                    resp = self.bottleneck[0].weight.detach() @ axis  # [hidden]
                    resp = resp / resp.norm().clamp(min=1e-8)
                    self.classifier.weight.copy_(resp.view(1, -1) * 3.0)
                    self.classifier.bias.fill_(1.5)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        attn = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        agg = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        logits = self.classifier(agg)
        return logits, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, warm_start=True)
