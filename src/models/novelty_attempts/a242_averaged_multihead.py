"""a242 - Averaged multi-head attention pooling. NEW: average (not concat) K attention reads -> diffuse.

Diagnostic showed uni2 over-concentrates (attention entropy 0.891, lowest) and sparsity-adders crash it.
a242 computes K=4 INDEPENDENT gated-attention distributions and AVERAGES them into one pooling weight
vector. Averaging diverse sharp heads yields a structurally MORE DIFFUSE read (an attention analogue of
ensembling), which should ease the uni2 over-concentration without a scalar knob. Distinct from a209
(multi-head CONCAT, which enriches but does not diffuse). Concept-free, self-contained,
permutation/size-invariant, deterministic, n=1 safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.W = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return F.softmax(self.W(self.V(h) * self.U(h)).squeeze(-1), dim=0)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, heads=4):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.heads = nn.ModuleList([Head(hidden_dim) for _ in range(heads)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = torch.stack([head(h) for head in self.heads], dim=0).mean(dim=0)  # average K attention dists
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
