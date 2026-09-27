"""a188 — Multi-statistic deep-sets pooling. NEW family: jointly pool 4 order-statistics, not one.

Instead of choosing ONE pooling, concatenate four complementary bag statistics of the bottleneck
features and let the head weigh them: attention-mean (salient), plain mean (holistic density), max
(strongest single patch), std (heterogeneity/spread of the fibre signal). Fibrosis grade plausibly
depends on BOTH overall density (mean) AND how uniformly dense (std) AND peak severity (max); a single
pooling collapses these. Richer than a165's attn+mean concat (adds max+std). Concept-free,
self-contained, permutation/size-invariant, deterministic at inference.
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
        self.classifier = nn.Linear(4 * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                       # [N,D]
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z_attn = torch.mv(h.t(), a)                         # salient
        z_mean = h.mean(dim=0)                              # holistic
        z_max = h.max(dim=0).values                         # peak
        z_std = h.std(dim=0, unbiased=False)                # spread
        z = torch.cat([z_attn, z_mean, z_max, z_std], dim=0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
