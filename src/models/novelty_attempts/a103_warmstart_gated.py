"""a103 — baseline gated-attention with a WARM-STARTED head (cheap probe).

The warm-start (head.weight initialised along the bottleneck response to the
train fibrosis axis) is the one ingredient that made the diffuse mean-pool clear
the seed=2 gate. The baseline gated-attention head is default-initialised. This
asks: does warm-starting the ACTUAL frontier model's head help PAIRED across
folds (a real, if small, win) — or does the init wash out under training (tie)?

a103 = ABMIL (identical aggregator) + head warm-started along the
seed=2 train fibrosis axis (init-only). Ablation a104 = warm_start=False = the
plain baseline. forward uses ONLY 'features'. RAW logits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_AXIS_PATH = Path(__file__).resolve().parents[3] / "data" / "prototypes_virchow2_reti_train_seed2.pt"


def _axis(input_dim: int) -> Optional[torch.Tensor]:
    try:
        v = torch.load(_AXIS_PATH, map_location="cpu", weights_only=False)["axis"].float().view(-1)
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
