"""a321 — Concentration-gated mahalanobis-leverage attention (a320 + adaptive gate).

DIRECTLY ATTACKS a320's failure mode. a320 suppressed high-leverage (outlier) patches uniformly: it
HELPED uni2 (removed junk from a concentrated read) but HURT titan/virchow2 (its high-leverage patches
ARE the informative spread-out density the diffuse grade needs). a321 GATES the suppression by the
attention CONCENTRATION: suppress outliers strongly only when the attention already forms a clear peaked
core (so the outliers are likely junk), and turn suppression OFF when the attention is diffuse (no clear
core -> every patch is plausibly informative, leave them alone). Hypothesis: uni2 (peaked) still gets
the de-junking benefit while titan/virchow2 (diffuse) are spared -> the disjoint-helper might break.

gate conc = 1 - 1/(N*sum a^2)  in [0,1) (0 = uniform/diffuse attention, ->1 = peaked).
e' = e - lam * conc * leverage_z.

DISCLOSURE: exploratory seed-2 fish (user's "keep searching, no multi-seed for now"); honest prediction
uncertain — the gate may not separate backbones the way hoped (per-bag signals were shown unreliable),
but it is a genuinely-new mechanism directly motivated by a320. ZERO extra params (197,250; lam fixed).
No ||h|| salience. Self-contained, permutation/size-invariant, deterministic eval, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, lam=0.5):
        super().__init__()
        self.lam = float(lam)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                                  # [N,H]
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)    # [N]
        N = h.shape[0]
        if N >= 2:
            a0 = F.softmax(e, dim=0)                                                    # pre-suppression weights
            kappa = (a0 * a0).sum()                                                     # in [1/N, 1]
            pr = 1.0 / (N * kappa + 1e-12)                                              # participation ratio (0,1]
            conc = (1.0 - pr).clamp(0.0, 1.0)                                           # 0=diffuse, ->1=peaked
            mu = h.mean(0)
            var = h.var(0, unbiased=False) + 1e-6
            lev = ((h - mu) ** 2 / var).sum(1)                                          # diag-Mahalanobis^2
            lev_z = (lev - lev.mean()) / (lev.std() + 1e-6)
            e = e - self.lam * conc * lev_z                                             # gated outlier suppression
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
