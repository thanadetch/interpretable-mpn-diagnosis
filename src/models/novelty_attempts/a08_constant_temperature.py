"""a08 — ablation companion to a07. Constant-temperature softmax (no 1/sqrt(N)).

Hint targeted: H4_length_normalised_softmean. Role: ablation companion.

Architecture is identical to a07_length_normalised_softmean.py with one
single difference:

    a07 (main):     tau = softplus(c_raw) / sqrt(N)   # length-normalised
    a08 (ablation): tau = softplus(c_raw)             # bag-size-independent

This isolates the *bag-size adaptive scaling* as the active ingredient.
If a07 beats a08, the 1/sqrt(N) is doing useful work (probably by
fixing the G0 outlier-fixation pattern). If a07 ~= a08, a learnable
global temperature is what matters; if both fail vs baseline, the
attention temperature is not where the gain lives.

Kill criterion: report regardless of outcome; exists only to contrast
    with a07.

Param count: baseline 197,250 + 1 scalar (c_raw) = 197,251.
    (Same as a07.)
"""
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,
        c_init: float = 1.0,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.c_raw = nn.Parameter(torch.tensor(_inv_softplus(c_init)))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_logits = self.attention_W(V * U).squeeze(-1)

        tau = F.softplus(self.c_raw)  # no 1/sqrt(N)
        attention = F.softmax(attn_logits * tau, dim=0)

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)
        y = self.classifier(bag)

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
    c_init=1.0,
)

