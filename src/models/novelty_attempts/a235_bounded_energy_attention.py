"""a235 - Bounded-energy gated attention (tanh-capped logits). Ablation companion to a234.

Same diagnostic motivation (uni2 over-peaks because its attention logits are high-contrast). Instead of
z-scoring, a235 SQUASHES the energies through a learnable soft cap: a = softmax(c * tanh(e / c)). This
bounds |logit| ~ c, so a backbone whose raw energies are large/high-contrast (uni2) cannot drive the
attention into an over-concentrated spike, while low-contrast backbones (virchow2) are nearly unchanged
(tanh is ~identity near 0). c is learnable -> the model picks the maximum allowed sharpness.
Concept-free, self-contained, permutation/size-invariant, deterministic, n=1 safe.
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
        self.log_c = nn.Parameter(torch.tensor(1.0))         # learnable soft cap on logit magnitude
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        c = F.softplus(self.log_c) + 0.5
        a = F.softmax(c * torch.tanh(e / c), dim=0)           # |logit| bounded by ~c -> caps sharpness
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
