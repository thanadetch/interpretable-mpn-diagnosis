"""a323 — Local-density coherence attention: up-weight patches in dense feature clusters.

DISTINCT from a320 (single-centroid mahalanobis distance, which mislabels multi-modal diffuse tissue as
outliers and hurt titan/virchow2). a323 measures each patch's LOCAL density = mean cosine similarity to
the other patches in the bag (a kernel/LOF-style coherence). Patches in dense clusters (coherent marrow
tissue, possibly several tissue modes) are gently UP-weighted; isolated patches (lone bone trabeculae /
fat / artefacts that sit alone in feature space) are gently DOWN-weighted. This is principle-aligned
(diffuse tissue forms clusters; bone/artefacts are isolated) and, unlike a320, does NOT punish a patch
for being far from a single global centroid -> multi-cluster diffuse tissue is preserved, which may spare
titan/virchow2 while still de-junking uni2.

e' = e + lam * density_z,   density_i = mean_j!=i cos(h_i, h_j).

DISCLOSURE: exploratory seed-2 fish (user's "keep searching, no multi-seed"). Keeps a320's winning
recipe (gentle, lam=0.5, continuous) but on a genuinely different statistic (local density vs centroid
distance). ZERO extra params (197,250; lam fixed). O(N^2) similarity, N<=~250 so MPS-cheap. No ||h||
salience (cosine is scale-free). Self-contained, permutation/size-invariant, deterministic, MPS-safe.
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
            hn = h / (h.norm(dim=1, keepdim=True) + 1e-6)                               # unit dirs
            S = hn @ hn.t()                                                             # [N,N] cosine sim
            density = (S.sum(1) - 1.0) / (N - 1)                                        # mean sim to others (excl self)
            density_z = (density - density.mean()) / (density.std() + 1e-6)
            e = e + self.lam * density_z                                               # up-weight high-local-density
        a = F.softmax(e, dim=0)
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
