"""asgap_scale - ASGAP with FIXED alpha and a FIXED bag-cardinality cap.

Base class for the sparsity-vs-bag-size study. It is the a215 (ASGAP) architecture with two
changes, both needed to ask one question:

    does the best attention sparsity depend on how many instances a bag contains?

    1. `alpha` is a fixed constructor argument instead of the (empirically inert) learnable
       parameter, so a run sits at exactly one point on the softmax -> sparsemax axis:
           alpha = 1.0  -> softmax   (dense; the ABMIL normalisation)
           alpha = 1.5  -> entmax    (the ASGAP default)
           alpha = 2.0  -> sparsemax (sparsest)
    2. `max_patches` caps every bag at N instances, applied in `forward` so it holds at BOTH
       train and eval time - the point is to simulate a different data regime, not to augment.
       The subset is drawn deterministically from the bag's own values, so a given bag always
       yields the same subset across epochs and between train and eval; resampling per epoch
       would be an augmentation and would confound the study.

Motivation: WSI-scale MIL reports the opposite of what this cohort shows - Attention Entropy
Maximization (MICCAI 2025) finds over-concentrated attention causes overfitting on bags of
thousands of patches, whereas sparse attention helps here on bags of ~44. If the optimal alpha
shifts with N, one rule explains both, and alpha=1.5 stops being a tuned constant.

    Peters et al., "Sparse Sequence-to-Sequence Models", ACL 2019 (alpha-entmax).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z: torch.Tensor, alpha: float, n_iter: int = 25) -> torch.Tensor:
    """alpha-entmax by bisection. alpha == 1 falls back to softmax (the limit case)."""
    if alpha <= 1.0 + 1e-6:
        return torch.softmax(z, dim=0)
    am1 = alpha - 1.0
    z = z * am1
    tau_lo = z.max() - 1.0
    tau_hi = z.max()
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        if p.sum() > 1:
            tau_lo = tau
        else:
            tau_hi = tau
    p = torch.clamp(z - tau_hi, min=0) ** (1.0 / am1)
    return p / p.sum().clamp(min=1e-8)


def deterministic_subset(features: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of a fixed pseudo-random k-subset, derived from the bag's own values.

    Seeding from the feature content (not from a global RNG) is what makes the subset stable
    across epochs, across train/eval, and across processes - so `max_patches` defines a data
    regime rather than acting as a stochastic augmentation.
    """
    n = features.shape[0]
    if k <= 0 or n <= k:
        return torch.arange(n, device=features.device)
    corners = (features[0, 0], features[0, -1], features[-1, 0], features[-1, -1])
    seed = n
    for c in corners:
        seed ^= int(abs(float(c)) * 1e6) & 0x7FFFFFFF
    g = torch.Generator()
    g.manual_seed(seed % (2**31 - 1))
    idx = torch.randperm(n, generator=g)[:k].sort().values
    return idx.to(features.device)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        alpha: float = 1.5,
        max_patches: int = 0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.max_patches = int(max_patches)  # 0 = no cap
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, features: torch.Tensor, return_attention: bool = False, metrics=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        if self.max_patches:
            features = features[deterministic_subset(features, self.max_patches)]
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, alpha=1.5, max_patches=0)
