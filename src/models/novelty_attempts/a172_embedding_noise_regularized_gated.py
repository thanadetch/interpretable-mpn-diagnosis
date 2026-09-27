"""a172 — Embedding-noise regularised gated attention (variance-reduction, targets the -0.95 trap).

LIGHT, self-contained. Keeps ABMIL EXACTLY and adds a single denoising regulariser: during
TRAINING ONLY, inject Gaussian noise into the bottleneck embeddings before attention/pooling. This
does NOT add capacity to fit the val cohort (the failure mode of all 187 prior priors); instead it
forces the head to rely on the diffuse, aggregate signal rather than memorising specific patch
embeddings -> a regulariser aimed directly at the documented val<->test (-0.95) instability.

    h_i      = bottleneck(x_i)
    h_i      = h_i + sigma * eps_i         (TRAIN ONLY; eps_i ~ N(0,I); off at inference)
    e_i      = W(V(h) * U(h)) ; a = softmax(e) ; z = Σ a_i h_i ; y = classifier(z)

sigma = exp(log_sigma) is LEARNABLE (init 0.1): the optimiser can shrink it to ~0 (=> exact baseline)
if noise does not help, so this is an honest, self-disabling regulariser (like a158's T). DETERMINISTIC
at inference (noise gated by self.training). Permutation- & size-invariant. 1 extra DOF. No external file.
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
        self.log_sigma = nn.Parameter(torch.tensor(math.log(0.1)))  # learnable noise std; can ->0

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        if self.training:
            sigma = self.log_sigma.exp()
            h = h + sigma * torch.randn_like(h)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        attn = F.softmax(e, dim=0)
        z = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, attn, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
