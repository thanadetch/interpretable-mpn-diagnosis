"""a333 — Regularised Content-Adaptive Learnable-Alpha ASGAP.

Diagnosis of the a327-a332 failures:
  * mode A (inert): a scalar alpha param gets too little gradient and never moves (a327/a330/a331).
  * mode B (overfit): a full content head (a328) DOES move alpha but the 257-dim descriptor lets it
    memorise the train set -> alpha drifts sparse (mean 1.67), val up but TEST DROPS.

a328 already PASSED val (0.8020) and missed test by only 0.006, so the head is close — it just
overfits. Standard fix = regularise the head while keeping it expressive enough to move:
  * keep the content head (avoids mode A);
  * apply Dropout(0.5) to the bag descriptor before the head (stops it memorising -> fixes mode B);
  * a mild anchor: alpha = 1.5 + 0.3*tanh(head) keeps it near the proven optimum but free to adapt.

If regularisation lets the head learn a *generalisable* (not memorised) per-bag alpha, val and test
could both hold. If it still fails, that is strong evidence the failure is the val<->test trap
itself (alpha adaptivity is parallel to it), not insufficient regularisation. desc = [mean h, std h,
z_N]. Fixed structure otherwise; permutation/size-invariant; deterministic eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

LOG_REF = 3.725
LOG_SCALE = 0.323
BAND = 0.30   # alpha in (1.2, 1.8)


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
        self.alpha_drop = nn.Dropout(0.5)          # regularise the alpha head's input
        self.alpha_head = nn.Linear(2 * hidden_dim + 1, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)       # alpha = 1.5 at start
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._last_alpha: Optional[float] = None

    def _alpha(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        z_n = torch.as_tensor((math.log(max(n, 1)) - LOG_REF) / LOG_SCALE,
                              dtype=h.dtype, device=h.device).view(1)
        desc = torch.cat([h.mean(0), h.std(0, unbiased=False), z_n], dim=0)
        desc = self.alpha_drop(desc)               # <-- dropout regularisation
        a = 1.5 + BAND * torch.tanh(self.alpha_head(desc).squeeze(-1))
        self._last_alpha = float(a.detach())
        return a

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        alpha = self._alpha(h)
        a = entmax_bisect(e, alpha)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
