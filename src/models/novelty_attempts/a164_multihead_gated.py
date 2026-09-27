"""a164 — Multi-head gated attention (light variance-reduction tweak of SimpleGatedMIL).

Keeps the ABMIL bottleneck + gated attention, but uses K attention SCORER heads that
share the V/U gating and differ only in the final 128→1 projection (so it stays light: K·129 extra
params). Each head yields its own attention distribution; the K attention-pooled vectors are
AVERAGED before the classifier — an in-model ensemble that reduces the attention variance that
plagues this small / unstable cohort. Self-contained (no axis/file).

Hypothesis: averaging K diverse attention reads is more stable than one head → slightly better
generalisation. Not grading-specific; tests whether the baseline's single attention head
under-uses capacity. K=4.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(n_heads)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        g = self.attention_V(h) * self.attention_U(h)
        zs = []
        attns = []
        for head in self.heads:
            a = F.softmax(head(g).squeeze(-1), dim=0)
            zs.append(torch.mv(h.t(), a))
            attns.append(a)
        z = torch.stack(zs, 0).mean(0)
        out = self.classifier(z).view(-1)
        if return_attention:
            return out, torch.stack(attns, 0).mean(0), None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
