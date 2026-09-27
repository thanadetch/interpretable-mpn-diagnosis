"""a216 - Hard mixture-of-pooling routing. NEW: each bag HARD-routes to ONE of K pooling experts.

A gate (from the bag descriptor) hard-selects (straight-through argmax at train, argmax at eval) ONE of
three pooling experts -- diffuse (high-T softmax), sparse (sparsemax), or mean -- per bag. Lets the model
use a different pooling for different bags discretely (vs a195's soft mix). Concept-free, self-contained,
deterministic at inference.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def sparsemax(z):
    zs, _ = torch.sort(z, descending=True)
    cssv = zs.cumsum(0) - 1.0
    k = torch.arange(1, z.shape[0] + 1, device=z.device, dtype=z.dtype)
    support = (zs - cssv / k) > 0
    km = support.sum().clamp(min=1)
    tau = cssv[km.long() - 1] / km
    return torch.clamp(z - tau, min=0.0)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.gate = nn.Linear(hidden_dim, 3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        n = h.shape[0]
        experts = torch.stack([
            torch.mv(h.t(), F.softmax(e / 3.0, dim=0)),     # diffuse
            torch.mv(h.t(), sparsemax(e)),                  # sparse
            h.mean(0),                                      # mean
        ], dim=0)                                           # [3,D]
        g = self.gate(h.mean(0))                            # [3]
        gh = F.softmax(g, dim=0)
        idx = torch.argmax(gh)
        onehot = F.one_hot(idx, 3).float() + gh - gh.detach()   # straight-through
        z = torch.mv(experts.t(), onehot)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, None, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
