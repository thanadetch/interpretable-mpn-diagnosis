"""a165 — Dual-pool (attention + mean) readout (light grading-motivated tweak of SimpleGatedMIL).

Motivated directly by the a158 finding: the optimal read is DIFFUSE-with-a-gentle-tilt — i.e. it
wants BOTH a holistic average AND a selective attention focus. So a165 gives the model both reads
explicitly: it concatenates the attention-weighted pooled vector with the plain mean-pooled vector
and lets the classifier weight them.

    z_attn = Σ softmax(e)_i · h_i          # selective gated read (baseline)
    z_mean = mean_i h_i                     # holistic / diffuse read (= mean pooling)
    y      = classifier([z_attn ; z_mean])  # use both

Self-contained (no axis/file). Light: classifier input doubles (128→256), +128 params. Grading
relevance: the holistic-density read (mean) and the selective read coexist, and the head decides
the balance — a data-driven version of "diffuse density + gentle emphasis".
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
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z_attn = torch.mv(h.t(), a)
        z_mean = h.mean(0)
        z = torch.cat([z_attn, z_mean], dim=0)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, a, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
