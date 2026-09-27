"""a201 - DropKey attention regularisation. NEW: regularise the ATTENTION, preserve feature signal.

a172 (feature noise) & a167 (patch dropout) were the ONLY mechanisms that ever passed titan GATE1 --
but both CRASHED G3 recall on weak backbones (they corrupt/drop the high-density feature signal that
G3 depends on). a201 regularises differently: DropKey (Li et al. 2022) randomly masks ATTENTION LOGITS
at train (set to -inf before softmax) so the model can't over-rely on a few patches, WITHOUT touching
the feature values -> aims to keep the titan-GATE1 regularisation benefit while preserving the G3
signal that killed GATE2. Off at inference (deterministic). Self-contained, concept-free.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, p_key=0.1):
        super().__init__()
        self.p_key = p_key
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        if self.training and self.p_key > 0:
            mask = torch.rand_like(e) < self.p_key
            if mask.all(): mask[torch.randint(0, e.shape[0], (1,))] = False
            e = e.masked_fill(mask, float("-inf"))
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
