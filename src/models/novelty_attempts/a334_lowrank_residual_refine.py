"""a334 — Fixed-Alpha ASGAP + Learnable Low-Rank Residual Refinement (non-alpha lever).

The 7 learnable-alpha designs (a327-a333) failed because alpha = sparsity is parallel to the
val<->test trap. This mechanism is ORTHOGONAL: keep entmax pooling at the proven fixed alpha=1.5
(unchanged from a215) and instead add a small LEARNABLE refinement to the *pooled representation*
quality — a different axis than pooling sharpness.

    z    = entmax_pool(h, alpha=1.5)              # 128-d, exactly a215
    r    = W2( relu( W1(z) ) )                    # low-rank residual, rank R=8
    z'   = z + gate * r                           # gate is a learnable scalar, init 0 -> identity
    y    = classifier(z')

gate initialised to 0 so the model starts byte-identical to a215 and only departs if the residual
demonstrably helps. Rank-8 bottleneck (128->8->128) keeps the added capacity small. This tests
whether *any* learnable capacity orthogonal to sparsity can improve on the tiny cohort; if it also
fails (inert gate = baseline, or overfit), that extends the negative result beyond the alpha axis.
Permutation/size-invariant, deterministic eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn

ALPHA = 1.5
RANK = 8


def entmax_bisect(z, alpha, n_iter=25):
    am1 = alpha - 1.0
    z = z * am1
    hi = z.max()
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
        # low-rank residual refinement on the pooled vector (orthogonal to pooling sparsity)
        self.refine = nn.Sequential(nn.Linear(hidden_dim, RANK), nn.ReLU(inplace=True),
                                    nn.Linear(RANK, hidden_dim))
        self.gate = nn.Parameter(torch.tensor(0.0))   # init 0 -> starts identical to a215
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._last_gate: Optional[float] = None

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, ALPHA)
        z = torch.mv(h.t(), a)
        z = z + self.gate * self.refine(z)            # learnable residual refinement
        self._last_gate = float(self.gate.detach())
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
