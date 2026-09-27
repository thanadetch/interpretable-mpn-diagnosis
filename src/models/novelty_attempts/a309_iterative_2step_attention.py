"""a309 — 2-step ITERATIVE gated-attention pooling (self-refinement); seed-2 fish #28.
DISCLOSURE: GATE2-at-seed-2 is PROVABLY UNREACHABLE by a drop-in (titan single-model optimum;
per-backbone α-optimum); this is a confirmatory candidate, expected to land at the noise floor.

Mechanism (genuinely distinct from prior candidates): do a first gated-attention pool to get a bag
descriptor z1, then RE-ATTEND over the patches using z1 as context — the second score combines the
gated-attention term with a dot-product similarity to z1 — and pool again to z2, which feeds the head.
Two hops of attention let the pool refine its focus given a first-pass summary, unlike the single-pass
baseline. Uses softmax (α=1) at both hops (uni2's optimum; titan tolerates it). DISTINCT from
single-pass gated attention (baseline/a215), consensus heads (a274/a283), ISAB/perceiver (a259/a261:
induced-point cross-attention, not self-refinement of the pooled vector), and ensembles. Self-contained,
permutation/size-invariant (scores depend on the set only through z1 and per-patch terms), deterministic
eval, MPS-safe. Modest extra capacity (one extra projection).
"""
from __future__ import annotations
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
        self.ctx = nn.Linear(hidden_dim, hidden_dim)        # projects z1 into the 2nd-hop query space
        self.scale = hidden_dim ** -0.5
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _gated_scores(self, h):
        return self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        e1 = self._gated_scores(h)
        a1 = F.softmax(e1, dim=0)
        z1 = torch.mv(h.t(), a1)                            # first-pass bag descriptor
        # second hop: gated score + dot-product similarity to z1 (context-conditioned re-attention)
        q = self.ctx(z1)                                    # [H]
        e2 = self._gated_scores(h) + self.scale * (h @ q)   # [N]
        a2 = F.softmax(e2, dim=0)
        z2 = torch.mv(h.t(), a2)
        y = self.classifier(z2).view(-1)
        if return_attention:
            return y, a2, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
