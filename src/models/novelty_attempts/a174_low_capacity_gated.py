"""a174 — Low-capacity gated attention (capacity-reduction test of the -0.95 trap escape).

LIGHT, self-contained. Identical to ABMIL but with a much SMALLER bottleneck (hidden_dim=32
vs 128). Direct hypothesis test: the documented val<->test (-0.95) instability is an OVERFITTING
phenomenon (a ~197K head drives val_qwk up on a 214-ROI / 10-patient val cohort without test gain).
If so, a head with far less capacity to overfit the val cohort should have its val-optimum coincide
more closely with the test-optimum -> more stable, possibly higher test. If it merely underfits, that
is itself an informative negative for the thesis (the instability is not pure capacity).

Same Ilse-style gated attention, scalar-regression head. No external file. ~12K params (vs ~197K).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        attn = F.softmax(e, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=32)
