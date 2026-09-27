"""a163 — ABMIL with a FIXED attention temperature (forces the diffuse mechanism ON).

Tests directly whether FORCING diffuse/holistic attention (instead of letting a158's temperature
collapse to T≈1) helps. The temperature is NOT learnable here — it is fixed to the env var
A163_T (default 2.0). T>1 flattens the attention toward uniform (= toward mean pooling as T→∞).

    attn = softmax(e / T)   with T fixed   (e = the usual gated-attention logits)

Hypothesis under test: if "grade = diffuse holistic density" then forcing larger T should help.
Prediction from the ablation (mean_pool < simple on ALL backbones): it will instead DEGRADE
monotonically toward the mean-pool number — i.e., selective attention beats holistic averaging on
these frozen features. Run a sweep over A163_T to trace the curve. Self-contained (no axis/file).
"""
from __future__ import annotations
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _fixed_T(default: float = 2.0) -> float:
    try:
        return float(os.environ.get("A163_T", default))
    except ValueError:
        return default


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.T = _fixed_T()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        attn = F.softmax(e / self.T, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, attn, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
