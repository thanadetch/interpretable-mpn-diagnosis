"""a331 — Dense-Side Learnable-Alpha ASGAP.

a328 made alpha genuinely learnable (content head, per-bag std ~0.2) but it drifted to the SPARSE
side (mean ~1.67) and test dropped. That leaves the opposite, untested hypothesis: maybe some bags
want to pool DENSER than 1.5 (toward softmax/mean) — more averaging, which on scarce data could be
more robust. a331 uses the same learnable content head as a328 (so alpha actually moves) but bounds
alpha to the DENSE side only:

    alpha = 1.5 - 0.45 * sigmoid( GAIN * head(desc) )     # alpha in (1.05, 1.5), start ~1.49

Head is zero-weight with bias -4 so alpha starts at ~1.49 (== a215) and can only be pulled denser as
the head learns. desc = [mean_n h, std_n h, z_N]. Everything else identical to a215/a328. This
directly tests the dense direction the sparse-drifting a328 never explored.
"""
from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn

LOG_REF = 3.725
LOG_SCALE = 0.323
GAIN = 2.0


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
        self.alpha_head = nn.Linear(2 * hidden_dim + 1, 1)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -4.0)   # sigmoid(-4)~0.018 -> alpha ~1.49 at start
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._last_alpha: Optional[float] = None

    def _alpha(self, h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        z_n = torch.as_tensor((math.log(max(n, 1)) - LOG_REF) / LOG_SCALE,
                              dtype=h.dtype, device=h.device).view(1)
        desc = torch.cat([h.mean(0), h.std(0, unbiased=False), z_n], dim=0)
        a = 1.5 - 0.45 * torch.sigmoid(GAIN * self.alpha_head(desc).squeeze(-1))  # (1.05, 1.5)
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
