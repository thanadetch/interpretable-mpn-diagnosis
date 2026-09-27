"""a328 — Content-Adaptive Learnable-Alpha ASGAP (CA-ASGAP).

Fixes the inert-alpha problem of a215 (and a327). In a215 the entmax order ``alpha`` is a single
global scalar squashed by sigmoid; it receives a weak, near-symmetric gradient and never leaves
~1.5 (init-sweeps a239/a240; a327 showed the same for a size-scalar). Root cause: a bare scalar is
disconnected from any strong, *input-varying* gradient signal, so its dataset-summed gradient
averages to ~0.

Fix: **predict alpha per bag from a small learnable head over a bag descriptor**, instead of a bare
scalar. The head's weights sit in the main gradient path AND are conditioned on the bag, so
different bags push the weights in different, non-cancelling directions -> the weights actually
learn, and alpha genuinely varies per ROI (content- and size-adaptive). This is the "make it
learnable" redesign.

    desc  = [ mean_n h , std_n h , z_N ]                 # 2*hidden + 1  (content + size)
    alpha = 1 + sigmoid( GAIN * alpha_head(desc) )        # in (1,2), per bag, LEARNABLE

Small-data prior kept via z_N (standardised log bag size, range 13..112 patches; few patches ->
head can learn to pool denser/averaged, many -> sparser/selective). Descriptor uses only
permutation-invariant statistics (mean/std over patches) so the model stays permutation- and
size-invariant; deterministic at eval. Head = Linear(2*hidden+1 -> 1): ~258 params over a215.

Distinct from prior art: a238 conditions a *temperature* on size; a269 adapts alpha to attention
*entropy*; a170 conditions a *temperature* per bag; a241 is FiLM on attention. None predicts the
entmax *alpha* from a bag descriptor. Concept-free, self-contained.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

LOG_REF = 3.725
LOG_SCALE = 0.323
GAIN = 2.0  # amplifies the head's effective gradient / output range


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
        # alpha predicted from a bag descriptor -> weights are in the gradient path -> learnable
        self.alpha_head = nn.Linear(2 * hidden_dim + 1, 1)
        nn.init.zeros_(self.alpha_head.weight)   # start at alpha=1.5 for every bag,
        nn.init.zeros_(self.alpha_head.bias)     # then let training move the head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._last_alpha: Optional[float] = None  # for introspection

    def _alpha(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        z_n = torch.as_tensor((math.log(max(n, 1)) - LOG_REF) / LOG_SCALE,
                              dtype=h.dtype, device=h.device).view(1)
        desc = torch.cat([h.mean(0), h.std(0, unbiased=False), z_n], dim=0)
        a = 1.0 + torch.sigmoid(GAIN * self.alpha_head(desc).squeeze(-1))  # in (1,2)
        self._last_alpha = float(a.detach())
        return a

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        alpha = self._alpha(h)                    # per-bag, content+size adaptive, learnable head
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
