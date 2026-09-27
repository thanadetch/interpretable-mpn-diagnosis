"""a209 - Multi-head gated pooling with CONCAT (not average). NEW: H independent pooled views concat.

a164 AVERAGED H attention heads (over a shared bottleneck). a209 keeps H heads' pooled vectors SEPARATE
and concatenates them, letting the head read multiple distinct saliency views jointly (different heads
can focus on different fibre aspects) instead of collapsing them. Concept-free, self-contained,
permutation/size-invariant, deterministic.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, H=4):
        super().__init__()
        self.H = H
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, H)         # H heads
        self.classifier = nn.Linear(H * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h))   # [N,H]
        a = F.softmax(e, dim=0)                                            # per-head softmax over patches
        z = (a.t() @ h).reshape(-1)                                       # [H*D] concat of head pools
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a.mean(dim=1), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
