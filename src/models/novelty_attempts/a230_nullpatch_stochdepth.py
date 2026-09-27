"""a230 - null-patch abstain (a220) + stochastic-depth (a208). Combo of two GATE2-reachers."""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, p_keep=0.5):
        super().__init__()
        self.p_keep = p_keep
        self.bottleneck = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.block = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.null_score = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        delta = self.block(self.ln(h))
        if self.training:
            if torch.rand(1).item() < self.p_keep: h = h + delta
        else:
            h = h + self.p_keep * delta
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        e_aug = torch.cat([e, self.null_score.view(1)], dim=0)
        a_aug = F.softmax(e_aug, dim=0); a = a_aug[:-1]
        z = torch.mv(h.t(), a) / a.sum().clamp(min=1e-6)
        y = self.classifier(z).view(-1)
        return (y, a, None) if return_attention else (y, None, None)


KWARGS = dict(input_dim=1280, num_classes=1)
