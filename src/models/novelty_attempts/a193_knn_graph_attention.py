"""a193 - kNN graph attention. NEW family: LOCAL graph message-passing (sparse), not global attention.

a184 used DENSE self-attention (every patch to every patch). a193 builds a sparse kNN graph in feature
space (each patch attends only to its k nearest neighbours), does one GAT-style message-passing step
to contextualise patches with their LOCAL fibre neighbourhood, then gated-pools. Local structure is the
relevant scale for a fibre meshwork. Concept-free, self-contained, permutation/size-invariant, det. eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, k=8):
        super().__init__()
        self.k = k
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        n = h.shape[0]
        k = min(self.k, n)
        sim = F.normalize(h, dim=1) @ F.normalize(h, dim=1).t()   # [N,N] cosine
        topv, topi = sim.topk(k, dim=1)                     # kNN incl self
        w = F.softmax(topv, dim=1)                          # [N,k] neighbour weights
        nbr = h[topi]                                       # [N,k,D]
        agg = (w.unsqueeze(-1) * self.msg(nbr)).sum(dim=1)  # [N,D] local message
        h2 = self.norm(h + F.relu(agg))                     # residual contextualise
        a = F.softmax(self.attention_W(self.attention_V(h2) * self.attention_U(h2)).squeeze(-1), dim=0)
        z = torch.mv(h2.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
