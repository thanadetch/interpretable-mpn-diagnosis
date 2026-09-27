"""a237 - Mean-prior robust blend of attention and mean pooling. NEW angle: target the val-variance regime.

The ceiling is variance-limited: val-selection on the 214-ROI cohort anti-correlates with test (~-0.95),
and higher-capacity / sharper reads overfit val. a237 anchors the bag read to the LOW-VARIANCE prior
(plain mean pooling) and lets it move toward attention only as far as training warrants:
    z = lambda * z_attn + (1 - lambda) * z_mean ,   lambda = sigmoid(lambda_raw), initialised near 0.
Starting at mean pooling (the robust baseline) and regularising toward it should reduce val-selection
variance. Concept-free, self-contained, permutation/size-invariant, deterministic, n=1 safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.lambda_raw = nn.Parameter(torch.tensor(-2.0))   # sigmoid(-2)=0.12 -> start near mean pooling
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z_attn = torch.mv(h.t(), a)
        z_mean = h.mean(dim=0)
        lam = torch.sigmoid(self.lambda_raw)
        z = lam * z_attn + (1.0 - lam) * z_mean
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
