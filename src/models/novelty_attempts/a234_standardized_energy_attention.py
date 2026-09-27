"""a234 - Standardized-energy gated attention. Diagnostic-driven (src/tools/diag_why_uni2.py).

The disjoint-helper failure traces to attention-logit CONTRAST differing per backbone: uni2's gated-
attention energies are high-contrast so softmax over-peaks (entropy 0.891) while virchow2's are low-
contrast/diffuse (0.965) -> no single sharpness fits all. a234 z-scores the per-bag energies BEFORE
softmax so logit contrast is normalised across backbones/bags, then a single LEARNABLE temperature T
sets one uniform sharpness on top. Idea: once contrast is backbone-invariant, one T may generalize.
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
        self.log_T = nn.Parameter(torch.tensor(0.0))         # learnable temperature on standardized logits
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        e = (e - e.mean()) / (e.std(unbiased=False) + 1e-5)  # per-bag z-score -> backbone-invariant contrast
        T = F.softplus(self.log_T) + 0.2
        a = F.softmax(e / T, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
