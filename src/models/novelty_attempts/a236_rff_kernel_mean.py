"""a236 - Random-Fourier-features kernel mean embedding pooling. NEW: pool the bag DISTRIBUTION in an RKHS.

Instead of an attention-weighted mean (a single first moment), map each bottleneck feature through fixed
random Fourier features phi(h) (approximating an RBF kernel) and mean-pool phi over patches. The result is
the kernel mean embedding of the bag's feature distribution (the MMD representation), which captures the
full distribution shape, not just its centroid. Distinct from covariance (a180) / Fisher (a194) / multistat
(a188). Permutation/size-invariant, deterministic at inference, n=1 safe. RFF projection is a fixed buffer
(seeded by the trainer's seed at construction, saved in the checkpoint).
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_rff=512, gamma=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.register_buffer("W_rff", torch.randn(hidden_dim, n_rff) * gamma)
        self.register_buffer("b_rff", torch.rand(n_rff) * 2 * math.pi)
        self.scale = math.sqrt(2.0 / n_rff)
        self.classifier = nn.Sequential(
            nn.Linear(n_rff, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                        # [N,H]
        phi = self.scale * torch.cos(h @ self.W_rff + self.b_rff)  # [N, n_rff] random Fourier features
        emb = phi.mean(dim=0)                                # kernel mean embedding [n_rff]
        y = self.classifier(emb).view(-1)
        if return_attention:
            # per-patch contribution to the embedding norm (for viz)
            w = (phi * emb).sum(dim=1).clamp(min=0)
            w = w / w.sum().clamp(min=1e-8) if float(w.sum()) > 0 else torch.full((h.shape[0],), 1.0 / h.shape[0])
            return y, w, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
