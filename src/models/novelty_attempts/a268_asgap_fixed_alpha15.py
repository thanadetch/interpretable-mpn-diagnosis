"""ASGAP with FIXED alpha=1.5 — fixed-alpha control for the proposed method.

Identical to a215 / ASGAP (Adaptive-Sparse Gated Attention Pooling) in EVERY way EXCEPT the Tsallis
alpha-entmax exponent is a hard-coded constant alpha=1.5 (there is NO learnable `alpha_raw` parameter).
a215's learnable alpha was found empirically INERT (stays at ~1.5 across all backbones/seeds), so this
fixed-1.5 model is expected to reproduce a215's results almost exactly — confirming that alpha=1.5 is the
operative design choice and that "learnable alpha" adds nothing. Concept-free, self-contained,
permutation/size-invariant, deterministic eval.

Note: `torch.tensor(0.0)` for a215's alpha_raw consumes no RNG, so removing it does NOT shift the random
init of the other layers — i.e. this model inits identically to a215 (same seed -> same V/U/W/heads).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

ALPHA = 1.5  # fixed Tsallis exponent (1=softmax, 2=sparsemax); 1.5 = intermediate adaptive-sparse


def entmax_bisect(z, alpha, n_iter=25):
    # generic alpha-entmax via bisection (Peters et al. 2019), alpha>1 — identical to a215
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max()
    lo = z.max() - 1.0
    tau_lo = lo; tau_hi = hi
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        Z = p.sum()
        if Z > 1: tau_lo = tau
        else: tau_hi = tau
    p = torch.clamp(z - tau_hi, min=0) ** (1.0 / am1)
    return p / p.sum().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # NO learnable alpha — alpha is fixed at ALPHA=1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, ALPHA)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
