"""a214 - Sorted-set GRU pooling. NEW family: read patches as a sequence ordered by saliency.

Sort patches by their gated-attention score (descending -> permutation-invariant), then run a small GRU
over that ordered sequence and take the final hidden state as the bag rep. This lets the aggregator
model the RANK PROFILE of saliency (how quickly signal falls off from the most- to least-salient patch),
which a weighted mean discards. Concept-free, self-contained, permutation-invariant (via sort),
deterministic at inference.
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
        self.score = nn.Linear(hidden_dim, 1)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        s = self.score(h).squeeze(-1)
        order = torch.argsort(s, descending=True)
        seq = h[order].unsqueeze(0)                         # [1,N,D] saliency-ordered
        out, _ = self.gru(seq)
        z = out[0, -1]                                      # final hidden state
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, F.softmax(s, dim=0), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
