"""a220 - Attention with a learned NULL patch. NEW: model can route weight to a 'nothing' sink.

Appends a learned null/background patch to the bag before softmax attention, so the model can down-
weight ALL real patches by sending attention mass to the null sink when no patch is salient -- a soft
'abstain' that prevents forcing weight onto weak patches. The pooled rep uses only the real patches'
renormalised weights. Concept-free, self-contained, permutation/size-invariant, deterministic.
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
        self.null_score = nn.Parameter(torch.tensor(0.0))    # learned null logit
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        e_aug = torch.cat([e, self.null_score.view(1)], dim=0)   # append null
        a_aug = F.softmax(e_aug, dim=0)
        a = a_aug[:-1]                                            # real-patch weights (may sum < 1)
        z = torch.mv(h.t(), a) / a.sum().clamp(min=1e-6)         # renormalise over real patches
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
