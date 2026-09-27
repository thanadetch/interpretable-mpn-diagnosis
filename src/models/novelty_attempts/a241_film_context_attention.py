"""a241 - FiLM content-modulated attention. NEW: condition attention sharpness on the bag's own content.

Lesson from this session: scalar attention-shape knobs (a215 alpha, a238 size-temp) converge INERT.
a241 instead modulates the pre-softmax attention energies by a FiLM transform whose (gamma, beta) are a
LEARNED FUNCTION of the bag context (mean bottleneck feature) -- a content-conditioned, higher-capacity
modulation, not a single global scalar. So a bag whose content calls for diffuse reading can be softened
and vice-versa, per bag, structurally. Targets the disjoint-helper (uni2 wants diffuse, virchow2 sharper).
Concept-free, self-contained, permutation/size-invariant, deterministic, n=1 safe.
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
        self.film = nn.Sequential(                            # bag context -> (gamma, beta)
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, 2))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                        # [N,H]
        ctx = h.mean(dim=0)                                  # bag context [H]
        gamma, beta = self.film(ctx)                         # content-conditioned modulation
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)  # [N]
        e = (1.0 + gamma) * e + beta                         # FiLM modulate sharpness+bias per bag
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
