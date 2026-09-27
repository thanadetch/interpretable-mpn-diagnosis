"""a320 — Mahalanobis-leverage robust attention: down-weight atypical (outlier) patches in the logit.

PATHOLOGY-GROUNDED (bone-avoidance prior): patches far from the bag's feature centroid (high statistical
leverage) are likely NON-tissue outliers — lone bone trabeculae, fat, stain/edge/section artefacts — that
the grade should NOT ride on. a320 computes a per-patch diagonal-Mahalanobis leverage and SUBTRACTS a
standardized leverage penalty from the gated-attention logit before softmax, so typical (central) patches
keep their weight and atypical ones are suppressed — a robust attention, distinct from trimming the pool
(a87) or shrinking the descriptor (a248) since it acts on the LOGIT location.

DISCLOSURE: this modifies the attention WEIGHTS, which is the proven uni2-fragile lever (every weight-
modifier so far crashes uni2's exact softmax-mean optimum). Honest prediction: likely crashes uni2 like
bucket-1 candidates. Run anyway under the user's explicit "just research, keep searching" request; disclose,
multi-seed-audit any pass. ZERO extra params (197,250; lambda fixed). No ||h|| salience (leverage is a
typicality statistic, not magnitude). Self-contained, permutation/size-invariant, deterministic, MPS-safe.
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
            mu = h.mean(0)                                                              # [H]
            var = h.var(0, unbiased=False) + 1e-6                                       # [H]
            lev = ((h - mu) ** 2 / var).sum(1)                                          # [N] diag-Mahalanobis^2
            lev_z = (lev - lev.mean()) / (lev.std() + 1e-6)                             # [N] standardized leverage
            e = e - self.lam * lev_z                                                    # suppress atypical patches
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
