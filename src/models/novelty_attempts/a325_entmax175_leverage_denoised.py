"""a325 — entmax-1.75 pool on leverage-DENOISED logits (a215 x a320 combination, genuinely novel).

Combines the two best per-backbone mechanisms (verified no disk module has both entmax AND a leverage/
mahalanobis term): a215's entmax-1.5 normalization (the diffuse backbones' optimum: titan/virchow2) and
a320's diagonal-Mahalanobis leverage penalty on the attention logit (the only thing that helped uni2 by
de-junking bone/artifact outliers). The hypothesis being TESTED: a320 helped uni2 by removing junk so the
read was cleaner; if the leverage penalty cleans the bag FIRST, the subsequent entmax might be less
uni2-fatal than plain entmax (a215, which crashed uni2 to 0.9262) while keeping the diffuse-backbone
benefit of sparsity.

e' = e - lam * leverage_z ;  a = entmax_bisect(e', alpha=1.75) ;  z = sum a_i h_i.

DISCLOSURE: exploratory seed-2 combination fish (user's "just research, keep searching"). Honest
prior: entmax still likely crashes uni2 (its softmax-mean optimum), so expect ~2/3 like a215; this tests
whether denoising rescues it. ZERO extra params (197,250; alpha & lam fixed). No ||h|| salience.
Self-contained, permutation/size-invariant, deterministic eval, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha=1.75, n_iter=25):
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
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, lam=0.5):
        super().__init__()
        self.lam = float(lam)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)    # [N]
        N = h.shape[0]
        if N >= 2:
            mu = h.mean(0)
            var = h.var(0, unbiased=False) + 1e-6
            lev = ((h - mu) ** 2 / var).sum(1)
            lev_z = (lev - lev.mean()) / (lev.std() + 1e-6)
            e = e - self.lam * lev_z                                                    # a320 denoise
        a = entmax_bisect(e, alpha=1.75)                                                 # a215 sparse pool
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
