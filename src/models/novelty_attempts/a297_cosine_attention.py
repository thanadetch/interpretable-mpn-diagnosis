"""a297 — Cosine (scale-invariant) attention scoring on the baseline softmax pool (seed-2 fish #18;
best-of-N, seed-2 only, NOT robust).

Single-axis magnitude perturbations are all mapped (baseline optimal on features/pooling/head/output).
a297 changes attention GEOMETRY instead: score patches by a LEARNED COSINE similarity to a query
direction (L2-normalise the bottleneck features for SCORING only, with a learnable temperature), so
attention depends on feature DIRECTION not magnitude. Patches are still pooled as the softmax-weighted
mean of the RAW features (uni2's optimal pool). Rationale: the feature-norm ‖h‖ is grade-uninformative
on this cohort (diagnostic Spearman ~0), so removing magnitude from the SCORING could make attention
cleaner without touching the (optimal) pooling. DISTINCT from ABMIL's gated MLP attention
(magnitude-sensitive) and from any norm-WEIGHTING idea (a45, ruled out) — here norm is REMOVED from
scoring, not used as a weight. Honest expectation: cosine vs gated attention is a geometry swap with no
prior signal it beats baseline test; recorded per the never-stop directive. Self-contained, perm/size-
invariant, deterministic eval, MPS-safe.
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
        self.query = nn.Linear(hidden_dim, hidden_dim)        # projects to scoring space
        self.q_dir = nn.Parameter(torch.randn(hidden_dim))    # learned query direction
        self.log_temp = nn.Parameter(torch.zeros(1))          # learnable temperature (init 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)
        q = F.normalize(self.query(h), dim=1)                 # direction-only patch keys
        d = F.normalize(self.q_dir, dim=0)                    # direction-only query
        e = (q @ d) * torch.exp(self.log_temp)                # cosine-sim scores * temperature
        a = F.softmax(e, dim=0)                                # baseline softmax pool over raw h
        z = torch.mv(h.t(), a)
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
