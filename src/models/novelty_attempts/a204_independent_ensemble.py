"""a204 - Internal independent ENSEMBLE. NEW lever: variance reduction at the MODEL level.

The -0.95 trap is a VARIANCE problem (val-selection on a 214-ROI cohort picks test-suboptimal points).
a204 attacks it directly: K fully-independent ABMIL sub-heads (separate bottleneck+attention
+classifier, different inits) whose scalar outputs are AVERAGED. Averaging decorrelated heads lowers
prediction variance -> potentially a more stable val-optimum that transfers to test. (a164 averaged
attention heads over a SHARED bottleneck; here the sub-models are fully independent = more decorrelated.)
Concept-free, self-contained, permutation/size-invariant, deterministic at inference.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.bottleneck(x)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        return self.classifier(torch.mv(h.t(), a)).view(-1)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.heads = nn.ModuleList([_Head(input_dim, hidden_dim, dropout) for _ in range(K)])

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        y = torch.stack([hd(features) for hd in self.heads], dim=0).mean(dim=0)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
