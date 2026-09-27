"""a184 — Set-Transformer block (inter-patch self-attention) then gated pool. NEW family.

All 197 priors pool patches INDEPENDENTLY (each patch scored on its own). a184 first lets patches
ATTEND TO EACH OTHER via one self-attention block (Set Transformer SAB: MHA + FFN, residual), so the
representation of each patch is contextualised by the rest of the bag BEFORE pooling. This models
patch-to-patch interaction (e.g. local fibre context) that independent attention cannot. Then the
usual gated-attention pool + linear head. Uses torch's MultiheadAttention (no new dependency).
Permutation-equivariant block + permutation-invariant pool; deterministic at inference (eval).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_heads=4):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.mha = nn.MultiheadAttention(hidden_dim, n_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        x = h.unsqueeze(0)                                  # [1,N,D]
        att, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm1(x + att)
        x = self.norm2(x + self.ff(x))
        h2 = x.squeeze(0)                                   # [N,D] contextualised
        a = F.softmax(self.attention_W(self.attention_V(h2) * self.attention_U(h2)).squeeze(-1), dim=0)
        z = torch.mv(h2.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
