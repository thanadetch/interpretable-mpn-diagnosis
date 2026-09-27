"""a287 — a286's sparse⊕diffuse dual descriptor, but combined by a LEARNABLE convex blend
(scalar lambda) instead of concat, then LayerNorm -> K=4 consensus head (uni2 seed-2 fish #8,
best-of-N, seed-2 only, NOT robust).

Diagnosis driving this: a286 (concat) cleared uni2+titan but DROPPED virchow2 val (the diffuse-mean
channel helped the over-concentrated backbones but hurt virchow2, whose high baseline val comes from
the SPARSE read). Fix: blend the two pooled descriptors with a single learnable scalar
lambda = sigmoid(blend) (init 0.5), so each backbone self-tunes its sparse-vs-diffuse mix —
virchow2 can learn lambda->0 (recover the a285-style sparse descriptor), uni2/titan can lean diffuse.
DISTINCT from a286 (fixed concat, head sees both) and from the a276-a278 family (those blended at the
attention-WEIGHT level w = mix of entmax/uniform; here the blend is over the POOLED descriptors with a
learnable gate, feeding the a285 LayerNorm consensus head). Single hidden_dim descriptor (not 2H).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha, n_iter=25):
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max() - (1.0 / am1) * 0.0
    lo = z.max() - 1.0
    tau_lo = lo; tau_hi = hi
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** (1.0 / am1)
        if p.sum() > 1: tau_lo = tau
        else: tau_hi = tau
    p = torch.clamp(z - tau_hi, min=0) ** (1.0 / am1)
    return p / p.sum().clamp(min=1e-8)


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        self.K = K
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.blend = nn.Parameter(torch.zeros(1))            # lambda = sigmoid(blend), init 0.5
        self.znorm = nn.LayerNorm(hidden_dim)
        self.P = nn.Linear(hidden_dim, K)
        self.G = nn.Parameter(torch.zeros(K))
        self.b = nn.Parameter(torch.zeros(K))
        self.read_w = nn.Parameter(torch.zeros(K))
        self.ALPHA = 1.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, self.ALPHA)
        z_sparse = torch.mv(h.t(), a)                        # sparse 1.5-entmax pool
        z_mean = h.mean(dim=0)                               # diffuse full-coverage pool
        lam = torch.sigmoid(self.blend)                      # learnable mix in (0,1)
        z = self.znorm((1.0 - lam) * z_sparse + lam * z_mean)
        u = self.P(z) * (1.0 + self.G) + self.b
        y = (F.softmax(self.read_w, dim=0) * u).sum().view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
