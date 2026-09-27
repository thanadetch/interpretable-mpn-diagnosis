"""a269 / EAES — Entropy-Adaptive Entmax Sparsity (builds on a215 / ASGAP).

a215 applies 1.5-entmax pooling with a single GLOBAL alpha (=1.5). The documented
"disjoint-helper" failure: sparsifying attention HELPS diffuse-attention backbones
(TITAN, normalised softmax-attention entropy H ~ 0.95) but HURTS the already-concentrated
one (UNI2-h, H ~ 0.89) by over-concentrating an already-peaked read.

EAES keeps a215 VERBATIM except the entmax exponent alpha is made a per-bag, closed-form,
monotone-increasing function of THAT bag's own softmax-attention entropy H:
    - diffuse bag (high H)        -> alpha -> 1.5   (== a215 exactly)
    - concentrated bag (low H)    -> alpha -> ~1     (softmax / dense; stop over-sparsifying)
This directly inverts the disjoint-helper trap. By construction it is near-identity to a215
on diffuse bags (so it inherits a215's TITAN behaviour), and on concentrated bags its read
falls back toward the plain-softmax baseline (its UNI2-h failure mode is floored at softmax,
it cannot bleed below baseline the way over-sparsified entmax does).

alpha is a deterministic function of the bag (NOT a learnable parameter) -> it cannot go inert.
ZERO new parameters vs a215. Self-contained, permutation/size-invariant, deterministic eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha, n_iter=25):
    # generic alpha-entmax via bisection (Peters et al. 2019), alpha>1 — copied VERBATIM from a215
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max() - (1.0 / am1) * 0.0
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
        # entropy->alpha schedule (FIXED python floats, NOT nn.Parameter -> cannot go inert)
        self.H_lo = 0.90        # below this (concentrated, UNI2-like) -> alpha = alpha_min (~dense)
        self.H_hi = 0.965       # above this (diffuse, TITAN-like)     -> alpha = 1.5 (== a215)
        self.s = 0.5            # alpha span: alpha in [1, 1.5] before clamp
        self.alpha_min = 1.02   # guard: entmax_bisect divides by (alpha-1); alpha->1 blows up 1/am1

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        N = h.shape[0]
        # normalised softmax-attention entropy of THIS bag (in [0,1]); permutation/size invariant
        p = F.softmax(e, dim=0)
        logN = torch.log(torch.tensor(float(max(N, 2)), device=e.device, dtype=e.dtype))
        H = -(p * torch.log(p + 1e-12)).sum() / logN
        frac = ((H - self.H_lo) / (self.H_hi - self.H_lo)).clamp(0.0, 1.0)
        alpha = (1.0 + self.s * frac).clamp(min=self.alpha_min)   # per-bag exponent in [alpha_min, 1.5]
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
