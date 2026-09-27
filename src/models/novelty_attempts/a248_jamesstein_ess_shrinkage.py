"""a248 - James-Stein shrinkage of the attention pool toward the bag mean, with an effective-sample-size noise term.

MECHANISM. Backbone is the locked gated-attention weighted pool (KEEP THE POOL):
    a = softmax( attention_W( attention_V(h) * attention_U(h) ) ),  z_attn = sum_i a_i g_i,
where g = value(h) (Linear(H,H)) puts the geometry in a learned coordinate frame -- the only added
capacity. The novelty sits ON TOP of the pool: shrink z_attn toward the bag mean mu = mean_i g_i by a
CLOSED-FORM, DATA-DERIVED James-Stein retention
    r = d2 / (d2 + sigma2 / n_eff + eps),
with d2 = ||z_attn - mu||^2, sigma2 = mean_i ||g_i - mu||^2 the within-bag dispersion, and
    n_eff = 1 / sum_i a_i^2  (inverse participation ratio = effective sample size of the attention weights).
Then z = mu + r * (z_attn - mu); y = classifier(z). The numerator sigma2/n_eff is the sampling variance of
the pooled estimator: a spiky read (tiny n_eff, e.g. uni2's over-concentrated attention) BLOWS UP the noise
term -> r small -> automatic shrink toward the robust mean (de-concentration); a diffuse, well-supported read
(large n_eff) gives r -> 1 and keeps its focus.

WHY NEW / NOT EXHAUSTED. r is fully determined by the bag's own statistics -- there is NO learnable
alpha/temperature/lambda/blend scalar anywhere, so nothing can go inert (the failure mode of a237's lone
lambda). It is NOT a243's per-bag sigmoid(MLP(mean)) content gate (an exhausted content-gated blend) and NOT
a221's lasso soft-threshold toward ZERO with a learnable lambda; here the anchor is the robust bag MEAN and
the mix coefficient is a James-Stein statistic of dispersion + attention effective-sample-size. The n_eff term
is the disjoint-helper remedy: it de-concentrates uni2 (tiny n_eff inflates the noise -> small r) WITHOUT any
global sharpness knob, and anchors spiky reads to the robust mean -> lowers val-selection read-variance.

CONSTRAINTS. Self-contained nn.Module; torch/nn/F only. MPS-safe: matmul (torch.mv), elementwise, softmax,
sum/mean, clamp -- no linalg/cdist/inverse/median. Permutation- & size-invariant; deterministic in eval();
n=1 safe (z_attn==mu==g0 -> d2=0, sigma2=0 -> r=0 -> z=mu=g0). No nn.Parameter scalars.
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
        self.value = nn.Linear(hidden_dim, hidden_dim)            # learned coordinate frame (only added capacity)
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                # [N,H]
        g = self.value(h)                                            # [N,H] learned coordinate frame
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)  # [N]
        a = F.softmax(e, dim=0)                                       # [N] sums to 1 -> KEEP THE POOL
        z_attn = torch.mv(g.t(), a)                                  # [H] attention-weighted pool
        mu = g.mean(dim=0)                                           # [H] robust bag mean
        n_eff = 1.0 / (a * a).sum().clamp(min=1e-6)                  # effective sample size in [1, N]
        diff = g - mu                                               # [N,H]
        sigma2 = (diff * diff).sum(dim=1).mean()                     # within-bag dispersion (scalar)
        d2 = ((z_attn - mu) ** 2).sum()                             # ||z_attn - mu||^2 (scalar)
        r = d2 / (d2 + sigma2 / n_eff + 1e-6)                        # James-Stein retention in [0,1]
        z = mu + r * (z_attn - mu)                                  # shrink pool toward bag mean
        y = self.classifier(z).view(-1)                             # shape (1,)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
