"""a312 — Learnable softmax<->sparse descriptor blend (clean-gradient per-backbone pooling shape).

MOTIVATION (direct response to a311's falsification + the disjoint-helper mechanism):
uni2's optimum is the SHAPE of softmax (exponential concentration); titan/virchow2 want a sparser
pool. A single FIXED pooling shape cannot satisfy all three (the disjoint-helper wall). BUT each
backbone is trained as a SEPARATE model, so a *learnable* blend weight lambda between a softmax pool
and a sparse (entmax-1.5) pool can, in principle, land at a DIFFERENT value per backbone purely from
that backbone's own training gradient — uni2 -> lambda~0 (stay softmax, its optimum), titan/virchow2
-> lambda>0 (gain sparsity) — WITHOUT any per-bag signal (the per-bag dispersion route was shown
uncorrelated with the backbone alpha-optimum, so it is avoided here).

WHY THIS MIGHT MOVE WHERE a215's LEARNABLE ALPHA DID NOT (honest disclosure of the key risk):
a215 parameterised alpha and ran it through the non-smooth entmax bisection threshold; the gradient
w.r.t. alpha was near-flat and alpha stayed INERT at its init (~1.5) on every backbone. Here, instead,
we compute BOTH pooled descriptors explicitly and convex-blend them at the DESCRIPTOR level:
    z = (1 - lambda) * z_softmax + lambda * z_sparse,   lambda = sigmoid(lambda_raw),
so the gradient is exactly  dz/dlambda = z_sparse - z_softmax  — a clean, directly-informative signal,
unlike the bisection-threshold path. The empirical question (the whole point of the run) is whether
this clean gradient lets lambda adapt per backbone, or whether it too sits inert. lambda is init at
0.5 (neutral) so the data decides the direction with no thumb on the scale.

DISCLOSURE: still a single backbone-agnostic drop-in; if lambda is inert this reduces to a fixed
intermediate blend (and intermediate-alpha was shown noisy, uni2 dips at alpha=1.25), so GATE2 is not
expected — but this is the one principled path left (per-backbone-by-training, not per-bag-by-signal)
and it is a clean test, not a pile-on. Does NOT weight by ||h|| (ruled out). Self-contained,
concept-free, permutation/size-invariant, deterministic eval, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def entmax_bisect(z, alpha=1.5, n_iter=25):
    # alpha-entmax via bisection (Peters et al. 2019), alpha>1; fixed alpha=1.5 (sparse pool)
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
        self.lambda_raw = nn.Parameter(torch.tensor(0.0))   # sigmoid -> 0.5 init (neutral blend)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)   # [N]
        a_soft = F.softmax(e, dim=0)
        a_sparse = entmax_bisect(e, alpha=1.5)
        z_soft = torch.mv(h.t(), a_soft)
        z_sparse = torch.mv(h.t(), a_sparse)
        lam = torch.sigmoid(self.lambda_raw)
        z = (1.0 - lam) * z_soft + lam * z_sparse
        y = self.classifier(z).view(-1)
        if return_attention:
            # report the effective attention (blended weights) for interpretability
            a = (1.0 - lam) * a_soft + lam * a_sparse
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
