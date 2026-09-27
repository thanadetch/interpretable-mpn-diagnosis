"""a314 — Dual-view pooling (softmax + entmax-1.5) concatenated, full linear head picks the combination.

MOTIVATION (builds on a313's durable lever + fixes a312's failure mode):
 - a313 showed: keeping the softmax pool present is uni2-SAFE BY CONSTRUCTION (uni2 not crashed).
 - a312 showed: blending softmax/sparse with a single learnable SCALAR lambda is INERT (lambda
   stayed ~0.5; one scalar has no per-backbone traction on this cohort).
a314 keeps BOTH pooled descriptors explicitly and concatenates them, then lets the FULL linear head
choose the combination:
    z = [ z_softmax ; z_sparse ] in R^{2H},   y = W z + b ,
i.e. y = W_soft . z_softmax + W_sparse . z_sparse + b. The combination is now the head's main trained
weight matrix (per-feature-dimension), NOT an inert scalar — so it has real per-backbone traction:
uni2 can drive W_sparse -> ~0 and recover the baseline softmax read (uni2-safe, cannot crash like the
weight-modifying pools a311/a312); the diffuse backbones titan/virchow2 can put weight on the sparse
(entmax-1.5) view, which is their known optimum (a215). One module covers both regimes because the
head, not a fixed pool shape, selects the mix.

DISCLOSURE: still a single backbone-agnostic drop-in. The head is Linear(2H -> 1) = ~2x the baseline
head capacity, which on the 214-ROI val cohort risks the same val-overfit / early-epoch selection that
made a313 fail the diffuse backbones (titan ep5, virchow2 ep1). GATE2 is therefore NOT expected, but
this is the principled test of "let the head, not a scalar, pick the pool" and is uni2-safe by
construction. No ||h|| weighting. Self-contained, concept-free, permutation/size-invariant,
deterministic eval, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha=1.5, n_iter=25):
    am1 = alpha - 1.0
    z = z * am1
    tau_lo = z.max() - 1.0
    tau_hi = z.max()
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        if p.sum() > 1: tau_lo = tau
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
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)   # consumes both pooled views

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)   # [N]
        a_soft = F.softmax(e, dim=0)
        a_sparse = entmax_bisect(e, alpha=1.5)
        z_soft = torch.mv(h.t(), a_soft)
        z_sparse = torch.mv(h.t(), a_sparse)
        z = torch.cat([z_soft, z_sparse], dim=0)                                      # [2H]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a_soft, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
