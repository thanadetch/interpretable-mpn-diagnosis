"""a167 — Patch-dropout regularized gated attention (train-time regularization, new lever).

A different lever from weighting tweaks: during TRAINING it randomly drops a fraction p of patches
before the (unchanged) gated attention, so the model cannot rely on a few specific patches and is
pushed to read the bag broadly/holistically — a regulariser for the small, unstable cohort. At
EVAL all patches are used (deterministic). Self-contained, no extra params, no axis/file.

    train: keep a random (1−p) subset of patches -> gated attention -> grade
    eval : all patches -> gated attention -> grade

Grading relevance (loose): forbidding reliance on individual patches encourages the diffuse,
bag-wide read the grading principle calls for; mainly it is a variance-reducing regulariser. p=0.15.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, p_patch=0.15):
        super().__init__()
        self.p_patch = p_patch
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        feats = features
        if self.training and feats.shape[0] >= 8 and self.p_patch > 0:
            keep = torch.rand(feats.shape[0], device=feats.device) >= self.p_patch
            if keep.sum() >= 4:
                feats = feats[keep]
        h = self.bottleneck(feats)
        a = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        z = torch.mv(h.t(), a)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, a, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
