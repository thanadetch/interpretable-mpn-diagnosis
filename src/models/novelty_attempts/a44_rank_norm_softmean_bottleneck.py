"""a44 — rank-norm soft-mean WITH ABMIL-style bottleneck (H21 ablation).

Philosophy bucket: attention_replace_parameter_free

Ablation companion to `a43_rank_norm_softmean.py`. Identical wiring
except a `Linear(input_dim, hidden_dim) + ReLU + Dropout(0.5)`
bottleneck is inserted before the classifier, and the rank-norm soft
weighting is applied over the *bottlenecked* features instead of the
raw frozen features.

Active ingredient under test
----------------------------
"no bottleneck" vs "bottleneck". Legacy a25 (= a43 here) projects the
raw 1280-d Virchow2 features directly through `Linear(1280, 1)` after
rank-weighted pooling. ABMIL (= the locked baseline) uses a
128-d bottleneck. a44 keeps the rank-norm pooling but adopts the
baseline's bottleneck capacity, so the a43 ↔ a44 contrast cleanly
isolates whether the bottleneck helps or hurts when paired with
rank-norm weighting on Virchow2.

Rank-norm weighting is applied AFTER the bottleneck, on h_i = ReLU(W h_i^raw),
because (a) bottlenecked feature norms are the relevant salience signal
once the projection is in place, and (b) computing the rank on the raw
features and then pooling bottlenecked features mixes two different
salience signals.

Predictions
-----------
- a44 > a43 → bottleneck is helpful; the win comes from the ABMIL
  capacity profile, not from the no-bottleneck direct-projection design.
- a43 > a44 → no-bottleneck direct projection is part of the legacy a25
  win; should be preserved in any future variant.
- a43 ≈ a44 → both are dominated by rank-norm weighting; bottleneck is
  capacity-neutral here.

Kill criterion
--------------
Family-level: if BOTH a43 AND a44 have val_qwk < 0.80 at seed=2,
declare H21 dead under the Virchow2 baseline.

Param count (input_dim=1280, hidden_dim=128)
--------------------------------------------
    bottleneck Linear(1280, 128) + bias  = 163,968
    classifier Linear(128, 1) + bias     =     129
    -------------------------------------------------
    total trainable                       = 164,097
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        tau: float = 8.0,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "a44 is scalar-regression only (num_classes=1)."
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        self.tau = tau
        self.clamp_output = clamp_output

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)  # [1, N, D]
        B, N, _ = features.shape

        # Bottleneck: per-patch projection to hidden_dim
        # Run as [B*N, D] then reshape back to keep batched semantics.
        h = self.bottleneck(features.reshape(B * N, -1)).reshape(B, N, -1)  # [B, N, H]

        # Rank patches by bottlenecked-feature norm (salience proxy)
        norms = h.norm(dim=-1)  # [B, N]
        order = norms.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1, order, torch.arange(N, device=features.device).expand(B, N)
        )
        w = torch.softmax(-ranks.float() / self.tau, dim=1)  # [B, N]

        # Weighted mean over bottlenecked features
        bag = (w.unsqueeze(-1) * h).sum(dim=1)  # [B, H]

        y = self.classifier(bag)  # [B, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if squeeze:
            y = y.squeeze(0)
            w = w.squeeze(0)
        return y, w, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau=8.0,
    clamp_output=True,
)

