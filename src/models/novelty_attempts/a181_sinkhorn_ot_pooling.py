"""a181 — Sinkhorn / optimal-transport pooling. STRUCTURALLY NEW family (transport, not attention).

Instead of softmax attention weights, compute the pooling weights as an optimal-transport plan between
the patches and a set of M learned reference "anchors" (prototypal fibre-pattern slots), via a few
Sinkhorn iterations (entropic OT). Each anchor pools a soft-assigned cluster of patches; the bag is
the concatenation of the M anchor-pooled vectors. This is a genuinely different aggregation geometry
(balanced transport with marginal constraints) vs the unconstrained softmax used by all 197 priors —
and is permutation/size-invariant and differentiable. Pure torch, NO new dependency.

    h     = bottleneck(features)               [N,128]
    cost  = -(h @ anchors^T)                   [N,M]   (similarity -> low cost)
    P     = Sinkhorn(cost, n_iters)            [N,M]   (rows ~ uniform over patches, cols balanced)
    z_m   = Σ_i P[i,m] h_i / Σ_i P[i,m]        [M,128] anchor-pooled vectors
    y     = classifier(flatten(z))             scalar

Concept-free, self-contained, deterministic at inference. Judged purely on the gates.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_anchors=4, n_iters=3, eps=0.5):
        super().__init__()
        self.M = n_anchors; self.n_iters = n_iters; self.eps = eps
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.anchors = nn.Parameter(torch.randn(n_anchors, hidden_dim) * (hidden_dim ** -0.5))
        self.classifier = nn.Linear(n_anchors * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                  # [N,128]
        n = h.shape[0]
        cost = -(F.normalize(h, dim=1) @ F.normalize(self.anchors, dim=1).t())  # [N,M] in [-1,1]
        K = torch.exp(-cost / self.eps)                                # [N,M]
        # Sinkhorn: row marginal uniform 1/N, col marginal uniform 1/M
        u = h.new_ones(n) / n
        for _ in range(self.n_iters):
            v = (1.0 / self.M) / (K.t() @ u).clamp(min=1e-8)           # [M]
            u = (1.0 / n) / (K @ v).clamp(min=1e-8)                    # [N]
        P = u.unsqueeze(1) * K * v.unsqueeze(0)                        # [N,M] transport plan
        Pn = P / P.sum(dim=0, keepdim=True).clamp(min=1e-8)            # column-normalise
        z = (Pn.t() @ h).reshape(-1)                                  # [M*128]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, P.sum(dim=1), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
