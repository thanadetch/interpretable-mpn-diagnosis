"""a189 - Attention on RAW frozen features (no 128-d bottleneck). NEW family: different info flow.

Every prior candidate first compresses to a 128-d bottleneck, then scores/pools there. a189 keeps the
pooling in the FULL frozen feature space: the gated-attention scorer reads the raw features and the
weighted average is taken over the RAW features, with a single linear head on top. Tests whether the
bottleneck compression (not the pooling) is what limits generalisation. Concept-free, self-contained,
permutation/size-invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = self.drop(features)
        e = self.attention_W(self.attention_V(x) * self.attention_U(x)).squeeze(-1)
        a = F.softmax(e, dim=0)
        z = torch.mv(x.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
