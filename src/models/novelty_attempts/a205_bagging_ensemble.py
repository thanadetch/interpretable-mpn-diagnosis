"""a205 - Bagging ensemble (patch-subsampled sub-heads). NEW: decorrelate heads via patch bagging.

Like a204 but each of K sub-heads sees a RANDOM 50% patch subset at train (bagging) so the heads
decorrelate more strongly -> stronger variance reduction. At inference all heads see the full bag and
average (deterministic). Bagging is the classic variance-reduction recipe; never applied as an in-model
MIL ensemble here. Concept-free, self-contained, deterministic at inference.
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
        n = features.shape[0]
        outs = []
        for hd in self.heads:
            if self.training and n >= 4:
                idx = torch.randperm(n, device=features.device)[: max(2, n // 2)]
                outs.append(hd(features[idx]))
            else:
                outs.append(hd(features))
        y = torch.stack(outs, dim=0).mean(dim=0)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
