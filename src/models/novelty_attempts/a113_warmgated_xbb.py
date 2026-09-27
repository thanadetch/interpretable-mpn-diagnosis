"""a113 — warm-started gated attention, BACKBONE-AGNOSTIC (cross-backbone breadth test).

Identical mechanism to a103 (SimpleGatedMIL + classifier head warm-started along
the seed=2 TRAIN fibrosis axis), but the axis file is selected by input_dim so
the SAME module runs on Virchow2 (1280), UNI2 (1536) and TITAN (768). a103
hardcodes the Virchow2 axis path and would silently fall back to plain on any
other backbone — a113 fixes that so the warm-start is actually applied per
backbone.

Purpose (advisor's "improvement must be broad, not one-model"): the warm-start
clears the seed=2 gate on Virchow2 (0.835/0.958 vs 0.789/0.952). Does the SAME
head-init help on UNI2 and TITAN too (broad), or only on Virchow2 (a single-model
lottery)? Pair a113 (warm) vs a114 (warm_start=False = plain SimpleGatedMIL) on
each backbone, all at seed=2 (axis + split both seed=2 -> train-only, no leak).

forward uses ONLY 'features'. RAW logits. Deterministic at inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA = Path(__file__).resolve().parents[3] / "data"
# seed=2 train-only fibrosis axis (c_G3 - c_G0), one per backbone, keyed by feature dim.
_AXIS_BY_DIM = {
    1280: _DATA / "prototypes_virchow2_reti_train_seed2.pt",
    1536: _DATA / "prototypes_uni2_reti_train_seed2.pt",
    768: _DATA / "prototypes_titan_reti_train_seed2.pt",
}


def _axis(input_dim: int) -> Optional[torch.Tensor]:
    path = _AXIS_BY_DIM.get(input_dim)
    if path is None or not path.exists():
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
