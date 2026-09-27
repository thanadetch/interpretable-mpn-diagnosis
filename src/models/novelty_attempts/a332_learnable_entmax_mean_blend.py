"""a332 — Learnable Entmax/Mean Blend ASGAP (small-data robust blend).

The 5 learnable-alpha attempts (a327-a331) all failed: the entmax order is best left FIXED at 1.5.
So a332 stops touching alpha (keeps it fixed 1.5 = a215) and instead makes a DIFFERENT thing
learnable — the balance between the sharp entmax-pooled vector and the robust mean-pooled vector:

    z_entmax = sum_i entmax(e, 1.5)_i * h_i     # sharp, selective (a215)
    z_mean   = mean_i h_i                       # robust, low-variance (good for scarce patches)
    g        = sigmoid( head([mean h, std h, z_N]) )   # learnable per-bag gate
    z        = g * z_entmax + (1 - g) * z_mean

Small-data rationale: for bags with few / noisy patches the mean is a lower-variance estimate; the
gate can learn to lean on it, while confident bags keep the selective entmax readout. Unlike the
alpha knob, the gate weights sit in the main gradient path (like a328, which DID move), and the
blend is a genuinely different mechanism from every alpha variant. Head zero-init + bias 0 -> g=0.5
at start (equal blend). Fixed alpha=1.5, permutation/size-invariant, deterministic eval.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

LOG_REF = 3.725
LOG_SCALE = 0.323
ALPHA = 1.5


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
        self.gate_head = nn.Linear(2 * hidden_dim + 1, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)     # g = 0.5 at start (equal entmax/mean blend)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._last_g: Optional[float] = None

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        n = h.shape[0]
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = entmax_bisect(e, ALPHA)
        z_entmax = torch.mv(h.t(), a)
        z_mean = h.mean(0)
        z_n = torch.as_tensor((math.log(max(n, 1)) - LOG_REF) / LOG_SCALE,
                              dtype=h.dtype, device=h.device).view(1)
        desc = torch.cat([h.mean(0), h.std(0, unbiased=False), z_n], dim=0)
        g = torch.sigmoid(self.gate_head(desc).squeeze(-1))
        self._last_g = float(g.detach())
        z = g * z_entmax + (1.0 - g) * z_mean
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
