"""a06 — ablation companion to a05. Constant (bag-independent) presence gate.

Hint targeted: H5_two_branch_presence_severity. Role: ablation companion.

Architecture is identical to a05_presence_severity_mult.py with one
single difference:

    a05 (main):     presence = sigma(Linear_p(bag))   # per-bag G0-vs->=G1 gate
    a06 (ablation): presence = sigma(gamma)           # bag-INDEPENDENT scalar

gamma is a single learnable parameter initialised to 0 (presence = 0.5
at the start; it can drift but cannot encode any per-bag signal).
Severity head (unbounded linear + offset) is preserved. This isolates
the *per-bag presence signal* as the active ingredient.

Kill criterion: report regardless of outcome; this run exists only to
    contrast with a05.

Param count: baseline 197,250 has one Linear(128->1) head (129).
    a06 keeps one severity head (129) and adds gamma (1).
    Net: +1 param. Total: 197,251.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,
        severity_offset: float = 1.5,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output
        self.severity_offset = float(severity_offset)

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        self.head_severity = nn.Linear(hidden_dim, num_classes)

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_scores = self.attention_W(V * U).squeeze(-1)
        attention = F.softmax(attn_scores, dim=0)

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)

        presence = torch.sigmoid(self.gamma)
        severity = self.head_severity(bag) + self.severity_offset
        y = presence * severity

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attention, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    severity_offset=1.5,
    gamma_init=0.0,
)

