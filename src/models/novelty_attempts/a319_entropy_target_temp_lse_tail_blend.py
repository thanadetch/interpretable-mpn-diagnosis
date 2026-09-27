"""a319 — Entropy-target temperature + LSE-tail blend, both gated off below a fixed entropy target.

GENUINELY NEW (disk has entmax-tau, never a bisection-on-TEMPERATURE root-find; a266 uses a LEARNABLE
log_tau + a [3H] concat readout = capacity blow-up). a319 combines two parameter-free corrections, BOTH
anchored to a fixed normalized-entropy target H* = 0.90 (just above uni2's measured ~0.891) so BOTH are
OFF on uni2 and ON only for over-diffuse titan/virchow2:
  (1) one-sided closed-form temperature solve: if the attention is too diffuse (H0 > H*), sharpen via a
      temperature T found by bisection so the normalized entropy hits H*; otherwise T = 1 (no-op).
  (2) a logsumexp "tail" readout (fixed tau=0.5) blended in by g = clamp((H0-H*)/0.1, 0, 1).
Single [H]-dim head (no concat → no a314-style capacity overfit). tau, H*, bounds are fixed constants.

uni2-safety: H0 <= H* ⇒ T=1 (exact baseline softmax-mean) AND g=0 (no lse) ⇒ bit-for-bit ABMIL.
DISCLOSURE: exploratory fish (user's "just research"); data ceiling may cap it; run, disclose, multi-seed
-audit any pass. ZERO extra params (197,250). Self-contained, concept-free, permutation/size-invariant,
deterministic eval, MPS-safe, no new deps.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.Hstar = 0.90
        self.tau = 0.5

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)    # [N]
        N = h.shape[0]
        if N < 2:
            a = F.softmax(e, dim=0)
            y = self.classifier(h[0]).view(-1)
            return (y, a, None) if return_attention else (y, None, None)

        logN = math.log(N)

        def Hn(T):
            p = F.softmax(e / T, dim=0)
            return -(p * torch.log(p + 1e-12)).sum() / logN

        H0 = Hn(1.0)                                                                    # tensor scalar (carries grad)
        if H0.item() <= self.Hstar:
            T = 1.0
        else:
            lo, hi = 0.05, 1.0
            for _ in range(20):
                Tm = (lo + hi) / 2.0
                if Hn(Tm).item() > self.Hstar:
                    hi = Tm
                else:
                    lo = Tm
            T = hi
        a = F.softmax(e / T, dim=0)
        z_attn = torch.mv(h.t(), a)                                                     # [H]
        lse = self.tau * (torch.logsumexp(h / self.tau, dim=0) - logN)                  # [H] smooth-max tail
        g = ((H0 - self.Hstar).clamp(min=0.0).clamp(max=0.1)) / 0.1                     # scalar in [0,1], grad-carrying
        z = (1.0 - g) * z_attn + g * lse
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
