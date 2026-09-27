"""a294 — FIXED 50/50 blend of the baseline softmax attention pool and the plain diffuse mean pool
(seed-2 fish #15; best-of-N, seed-2 only, NOT robust).

uni2 favors density (a290: its sparsity optimum is alpha=1=softmax; sparser hurts). a294 pushes
slightly MORE diffuse than the baseline by pooling z = 0.5*(softmax-attention-mean) + 0.5*(plain mean),
a FIXED convex blend (non-learnable, so it cannot be inert like a287's learned lambda which froze at
0.5). Plain mean = maximally diffuse full-coverage density (matches the grading principle). Single Linear
head untouched. DISTINCT from a287 (learnable blend), a278 (entmax⊕mean), and MeanPoolMIL/ABMIL
(pure endpoints). Honest expectation: lands between ABMIL (uni2 0.9418) and MeanPool (worse),
so likely <= baseline; recorded per the never-stop directive. Self-contained, perm/size-invariant,
deterministic eval, MPS-safe.
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
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)
        z_att = torch.mv(h.t(), a)                            # softmax attention pool
        z_mean = h.mean(dim=0)                                # plain diffuse mean
        z = 0.5 * z_att + 0.5 * z_mean                        # FIXED 50/50 blend (denser-leaning)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
