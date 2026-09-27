"""a173 — Input-feature-noise regularised gated attention (denoise the FROZEN embeddings).

LIGHT, self-contained. Sibling of a172 but the noise is injected at the INPUT (frozen backbone
features) rather than the bottleneck output -> it also regularises the bottleneck Linear, a slightly
stronger denoising point. Same rationale: a variance-reduction regulariser aimed at the documented
val<->test (-0.95) instability, NOT a new val-fitting prior.

    x_i      = x_i + sigma * eps_i          (TRAIN ONLY; eps_i ~ N(0,I); off at inference)
    h_i      = bottleneck(x_i)
    e_i      = W(V(h) * U(h)) ; a = softmax(e) ; z = Σ a_i h_i ; y = classifier(z)

sigma = exp(log_sigma) is LEARNABLE (init 0.1) and can shrink to ~0 (=> exact baseline) if noise does
not help (self-disabling). DETERMINISTIC at inference. Permutation- & size-invariant. 1 extra DOF.
"""
from __future__ import annotations
import math
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
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.1)))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        x = features
        if self.training:
            x = x + self.log_sigma.exp() * torch.randn_like(x)
        h = self.bottleneck(x)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        attn = F.softmax(e, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
