"""a244 - DPP-inspired redundancy-repulsion pooling (MPS-native approximation).

INTENT (from the DPP marginal-inclusion idea): weight a patch by its quality q_i but DOWN-weight it when it
is redundant with OTHER high-quality patches -- so near-duplicate over-concentrated patches (uni2's failure
mode) repel each other and de-concentrate, while unique relevant patches keep weight. One mechanism that
moves opposite directions per backbone, targeting the disjoint-helper wall; inclusion-like weights are
bounded/saturating, which caps any single fluke patch and lowers val-selection variance.

IMPLEMENTATION NOTE (honest): the EXACT DPP marginal kernel K = L (L+I)^{-1} needs a linear solve / eigh,
and neither torch.linalg.solve (its backward) nor torch.linalg.eigh is supported on the Apple-MPS trainer
(verified: solve-backward errors on lu pivots, eigh unimplemented). So this uses a MATMUL-ONLY, MPS-native
surrogate of the inclusion probability -- a DPP-INSPIRED repulsion, NOT the exact determinantal marginal:
    q_i = softplus(relevance(h_i)) >= 0            # quality (full linear head, not a lone shape-scalar)
    S   = relu(<z_i, z_j>),  z = h/||h||           # cosine similarity, S_ii = 1
    r_i = (S q)_i - q_i = sum_{j!=i} S_ij q_j       # redundancy: quality-weighted similarity to OTHERS
    p_i = q_i / (1 + r_i)                           # inclusion-like: high quality + low redundancy -> high
    a   = p / sum(p)                                # pooling weights (sum 1)
Concept-free, self-contained (torch/nn/F only), permutation/size-invariant, deterministic, n=1 safe
(r=0 -> p=q -> a=[1]).
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
        self.relevance = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                        # [N,H]
        q = F.softplus(self.relevance(h).view(-1)) + 1e-4    # [N] quality >= 0
        z = h / (h.norm(dim=1, keepdim=True) + 1e-6)         # unit rows
        S = F.relu(z @ z.t())                                # [N,N] similarity, S_ii=1
        r = (S @ q) - q                                      # [N] redundancy from OTHER patches
        p = q / (1.0 + r.clamp_min(0.0))                     # [N] inclusion-like weight
        a = p / (p.sum() + 1e-8)                             # [N] sums to 1
        bag = a @ h                                          # [H]
        y = self.head(bag).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
