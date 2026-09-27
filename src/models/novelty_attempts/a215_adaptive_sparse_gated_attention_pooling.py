"""ASGAP — Adaptive-Sparse Gated Attention Pooling (thesis method name; experiment id a215).

Ilse-style gated attention with the softmax normalisation replaced by Tsallis alpha-entmax
(Peters et al. 2019): alpha=1 -> softmax (dense), alpha=2 -> sparsemax (sparsest). alpha is
*implemented* as a learnable parameter in [1,2] via a bisection projection, but was found
empirically INERT (it stays at its initialisation ~1.5; see init-sweep a239/a240) -> in practice
this is fixed adaptive-sparse pooling at alpha=1.5. "Adaptive" = the entmax threshold tau is solved
per bag, so the *number* of non-zero-weight patches adapts to each bag's score distribution (not a
fixed top-k). Concept-free, self-contained, permutation/size-invariant, deterministic eval.

(Historical note: the module was originally named `a215_learnable_entmax`; that name is kept as a
backward-compat alias for ~43 frozen experiment configs. "learnable" is a misnomer — alpha is inert.)
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha, n_iter=25):
    # generic alpha-entmax via bisection (Peters et al. 2019), alpha>1
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
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))   # sigmoid->0.5 -> alpha=1.5 init
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        alpha = 1.0 + torch.sigmoid(self.alpha_raw)        # in (1,2)
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
