"""a231 - Coverage-histogram pooling along a LEARNED density axis. NEW family: no attention-weighted mean.

Grading principle: reticulin grade = diffuse, bag-wide fibre DENSITY, i.e. what FRACTION of the tissue
is dense (coverage), not a few salient patches. This aggregator drops attention pooling entirely: it
projects each patch onto a single learned density axis, normalises per bag (size-invariant), then builds
a SOFT HISTOGRAM (coverage profile) of how patches distribute over K density bins, and grades from that
profile. Encodes the distribution of density across the bag, the data-derived-fibrosis-axis direction in
NOVELTY_AGENT_NOTES C1. Concept-free, self-contained, permutation/size-invariant, deterministic, n=1 safe.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, n_bins=12):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.density = nn.Linear(hidden_dim, 1)              # learned 1-D fibrosis/density axis
        self.register_buffer("centers", torch.linspace(-2.0, 2.0, n_bins))
        self.log_tau = nn.Parameter(torch.tensor(0.0))       # learnable soft-bin width
        self.classifier = nn.Sequential(
            nn.Linear(n_bins, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_classes))

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                        # [N,H]
        s = self.density(h).squeeze(-1)                      # [N] density score
        s = (s - s.mean()) / (s.std(unbiased=False) + 1e-5)  # per-bag normalise (size-invariant)
        tau = F.softplus(self.log_tau) + 0.2
        m = F.softmax(-(s[:, None] - self.centers[None, :]) ** 2 / tau, dim=1)  # [N,K] soft bin membership
        hist = m.mean(dim=0)                                 # [K] coverage profile (fraction per density bin)
        y = self.classifier(hist).view(-1)
        if return_attention:
            a = m.max(dim=1).values                          # per-patch peak membership (for viz)
            return y, a / a.sum().clamp(min=1e-8), None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
