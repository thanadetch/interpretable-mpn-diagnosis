"""a180 — Second-order (covariance) pooling. STRUCTURALLY NEW family (not a first-order tweak).

All 197 prior candidates pool to a FIRST-ORDER summary (attention-weighted MEAN of features). a180
pools the attention-weighted COVARIANCE instead: reticulin fibrosis is a MESHWORK TEXTURE, and
second-order statistics (feature co-occurrence) capture texture/structure that a first-order mean
cannot. The bag is represented by both its weighted mean AND the upper-triangle of its weighted
feature covariance.

    h   = bottleneck(features)           [N, 128]
    p   = project(h)                     [N, k]      (k=16, keeps cov small: k(k+1)/2 = 136)
    a   = softmax(gated_attention(h))    [N]         (same Ilse gating as baseline)
    mu  = Σ a_i p_i                                   weighted mean        [k]
    C   = Σ a_i (p_i-mu)(p_i-mu)^T                    weighted covariance  [k,k]
    z   = [ mu , vec(upper-tri(C)) ]                 [k + k(k+1)/2]
    y   = classifier(z)

Concept-free, self-contained, permutation- & size-invariant, deterministic at inference. Not
baseline-init (it is a different architecture, not a tweak) — judged purely on the gates.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, k=16):
        super().__init__()
        self.k = k
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.project = nn.Linear(hidden_dim, k)
        self._triu = torch.triu_indices(k, k)
        zdim = k + (k * (k + 1)) // 2
        self.classifier = nn.Linear(zdim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        a = F.softmax(e, dim=0)                       # [N]
        p = self.project(h)                           # [N,k]
        mu = torch.mv(p.t(), a)                       # [k]  weighted mean
        d = p - mu.unsqueeze(0)                       # [N,k]
        C = (d * a.unsqueeze(1)).t() @ d              # [k,k] weighted covariance
        ti, tj = self._triu.to(C.device)
        z = torch.cat([mu, C[ti, tj]], dim=0)         # [k + k(k+1)/2]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
