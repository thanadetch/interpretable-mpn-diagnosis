"""a327 — Size-Adaptive Sensitive-Alpha ASGAP (SA-ASGAP).

Motivation (small-data ROI): ASGAP (a215) makes the entmax sparsity order ``alpha`` learnable
but it is empirically INERT — it sits at ~1.5 regardless (init-sweep a239/a240) because a single
global scalar squashed through ``sigmoid`` receives a tiny, near-symmetric gradient. Two changes
here target the *limited-data-per-ROI* regime:

1. **Sensitive** — amplify the alpha gradient with a fixed GAIN and, more importantly, tie alpha to
   a real per-bag signal so it has a reason to vary (a stuck global scalar cannot adapt).
2. **Size-adaptive (the small-data prior)** — condition alpha on the bag size N (number of tissue
   patches; range 13..112 here, median 40). Small ROIs (few patches) → *lower* alpha → denser,
   more mean-like pooling that averages over the scarce evidence (lower variance, overfit-safer);
   large ROIs → *higher* alpha → sparser, more selective pooling that can focus on fibrotic patches
   without diluting. The direction is a learnable coefficient ``w_size`` (data can overrule the
   prior), initialised positive to encode "more patches -> allow more sparsity".

    z_N   = (log N - LOG_REF) / LOG_SCALE            # standardised bag size (dataset constants)
    alpha = 1 + sigmoid( GAIN * (a_raw + w_size * z_N) )   in (1, 2), per bag

Distinct from prior art in the repo: a238 conditions a softmax *temperature* on size (not the
entmax alpha); a269 adapts alpha to attention *entropy* (not bag size); a239/a240 only sweep a
fixed global alpha init. Only +1 scalar param over a215 (w_size) -> small-data safe.
Concept-free, self-contained, permutation-invariant, deterministic at eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

# dataset bag-size constants (log N over all 1330 reti bags; identical across backbones/tiling)
LOG_REF = 3.725
LOG_SCALE = 0.323
GAIN = 3.0  # amplifies alpha's gradient so it is no longer inert


def entmax_bisect(z, alpha, n_iter=25):
    # generic alpha-entmax via bisection (Peters et al. 2019), alpha>1
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
        # learnable global level (a_raw) + learnable size-sensitivity (w_size)
        self.a_raw = nn.Parameter(torch.tensor(0.0))    # base alpha 1.5 at median bag size
        self.w_size = nn.Parameter(torch.tensor(0.5))   # prior: bigger bag -> sparser
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _alpha(self, n_patches: int) -> torch.Tensor:
        z_n = (math.log(max(n_patches, 1)) - LOG_REF) / LOG_SCALE
        z_n = torch.as_tensor(z_n, dtype=self.a_raw.dtype, device=self.a_raw.device)
        return 1.0 + torch.sigmoid(GAIN * (self.a_raw + self.w_size * z_n))  # in (1,2)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        alpha = self._alpha(features.shape[0])          # per-bag, size-adaptive
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
