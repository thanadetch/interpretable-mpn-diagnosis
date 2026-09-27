"""a270 / TCFE — Two-stage Coarse-Fine Entmax pooling (builds on a215 / ASGAP).

a215 applies 1.5-entmax DIRECTLY over the N patch scores -> it can over-sparsify an already
concentrated read (the disjoint-helper failure on UNI2-h). TCFE moves the sparsity OFF the
patch axis: it soft-assigns patches to K=4 learned regions, pools DENSELY (membership-masked
softmax over patches) within each region to get K region summaries, then applies 1.5-entmax
ONCE over the K region summaries (coarse, sparse over the small fixed K-axis). So "where to
look" can be sparse (over K regions) while "how to summarise there" stays dense (over patches)
-> no per-patch zeroing to fight, UNI2-h's natural concentration is preserved.

a215 is the K=1 degenerate case, so TCFE strictly generalises it. alpha is FIXED at 1.5 (a
python const, NOT a learnable parameter — the learnable-alpha lesson from a215).
Self-contained, permutation/size-invariant, deterministic eval. Concept-free; no zero-shot
labels; no feature-norm weighting.
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
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.region_centers = nn.Parameter(torch.randn(K, hidden_dim) * (hidden_dim ** -0.5))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.region_score = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                          # [N,H]
        m = F.softmax(h @ self.region_centers.t(), dim=1)                      # [N,K] soft memberships
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)  # [N]
        logm = torch.log(m.clamp(min=1e-8))                                    # [N,K]
        a_fine = F.softmax(e.unsqueeze(1) + logm, dim=0)                       # [N,K] DENSE within-region over patches
        r = a_fine.t() @ h                                                     # [K,H] region summaries
        g = self.region_score(r).squeeze(-1)                                   # [K] region scores
        b = entmax_bisect(g, 1.5)                                              # [K] COARSE sparse over regions
        z = b @ r                                                              # [H] bag descriptor
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, (a_fine @ b), None                                       # [N] effective per-patch weight
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
