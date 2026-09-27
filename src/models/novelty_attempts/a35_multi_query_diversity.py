"""a35 - K=4 multi-query cross-attn + query orthogonality penalty (H18, batch 18 main).

Adds a query-orthogonality regulariser via a backward hook on the
learnable queries Q, so no trainer modification is needed.

L_div = || Q_norm @ Q_norm.T - I ||_F^2 / (K * (K-1))
hook:   grad(Q) += lambda_div * d(L_div)/d(Q)   at each backward pass.

Hypothesis: batch 17 K=8 val drop suggests query redundancy. Forcing
the K queries to be orthonormal in 64-d space should make K=4 more
stable (less collapse onto similar directions) and possibly push val
above a29 0.7997.

Ablation companion: a36_multi_query_no_diversity (lambda_div=0; bit-
for-bit positive control for a29).

Kill criterion: abandon H18 if a35 val < a29 0.7997 at seed=2.

Param count: same as a29 = 172,993 (hook adds no parameters).
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        query_dim: int = 64,
        n_queries: int = 4,
        dropout: float = 0.5,
        clamp_output: bool = True,
        query_init_std: float = 0.02,
        lambda_div: float = 0.1,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert n_queries >= 1

        self.n_queries = n_queries
        self.scale = query_dim ** -0.5
        self.clamp_output = clamp_output
        self.lambda_div = float(lambda_div)

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.key_proj = nn.Linear(hidden_dim, query_dim)
        self.queries = nn.Parameter(torch.randn(n_queries, query_dim) * query_init_std)
        self.classifier = nn.Linear(n_queries * hidden_dim, num_classes)
        nn.init.constant_(self.classifier.bias, 1.5)

        if self.lambda_div > 0 and n_queries > 1:
            self.queries.register_hook(self._diversity_grad_hook)

    def _diversity_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            Q = self.queries.detach().clone().requires_grad_(True)
            Q_norm = Q / Q.norm(dim=1, keepdim=True).clamp(min=1e-8)
            G = Q_norm @ Q_norm.t()
            I = torch.eye(self.n_queries, device=Q.device, dtype=Q.dtype)
            off = G - I
            denom = max(self.n_queries * (self.n_queries - 1), 1)
            L_div = (off ** 2).sum() / denom
            (g_div,) = torch.autograd.grad(L_div, Q)
        return grad + self.lambda_div * g_div

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        k = self.key_proj(h)
        scores = self.queries @ k.t() * self.scale
        attn = F.softmax(scores, dim=1)
        bag = (attn @ h).reshape(-1)
        y = self.classifier(bag.unsqueeze(0)).squeeze(0)
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)
        if return_attention:
            return y, attn.mean(dim=0), None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    query_dim=64,
    n_queries=4,
    dropout=0.5,
    query_init_std=0.02,
    lambda_div=0.1,
)
