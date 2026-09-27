"""a137 — TRIMMED-MEAN per-patch score pooling (user's idea: cut head & tail, keep the middle).

Same per-patch latent severity head as PerPatchScorePoolingMIL (MLP: LayerNorm ->
Linear(D->h) -> GELU -> Dropout -> Linear(h->1)), but the bag score is a SYMMETRIC
TRIMMED MEAN of the per-patch scores: sort the N patch scores, DROP the top `trim`
fraction and the bottom `trim` fraction, average only the MIDDLE.

Pathology rationale (grading principle): grade = diffuse, bag-wide reticulin density.
Bone trabeculae / background / stain-edge tiles are OUTLIERS that land at the extreme
high/low tail of the per-patch severity distribution; trimming removes them by
construction and keeps the central (typical) tissue density. A trimmed mean has a
higher breakdown point than the plain mean (robust to outlier patches) while staying
diffuse (no spiking on a few patches) — the opposite of top-k/quantile, which CHASE
the extreme tail.

trim fraction is read from env A137_TRIM (default 0.10 = drop 10% each side, keep
middle 80%). A137_TRIM=0 reproduces the plain mean (the clean ablation).
Always keeps at least 1 patch. forward matches the MIL signature: returns
(bag_score, patch_scores, None). Deterministic, permutation- & size-invariant.
"""
from __future__ import annotations
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _trim() -> float:
    try:
        t = float(os.environ.get("A137_TRIM", "0.10"))
    except ValueError:
        t = 0.10
    return min(max(t, 0.0), 0.49)


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        use_mlp_head: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1
        self.trim = _trim()
        if use_mlp_head:
            self.score_head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.score_head = nn.Linear(input_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze_back = False
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]
            squeeze_back = True
        B, N, _ = features.shape
        patch_scores = self.score_head(features).squeeze(-1)  # [B, N]

        lo = int(math.floor(self.trim * N))
        hi = N - lo
        if hi - lo < 1:               # degenerate (tiny bag / large trim) -> keep all
            lo, hi = 0, N
        sorted_scores, _ = patch_scores.sort(dim=1)        # ascending
        kept = sorted_scores[:, lo:hi]                      # [B, hi-lo] middle band
        bag_score = kept.mean(dim=1, keepdim=True)          # [B, 1] trimmed mean

        if squeeze_back:
            bag_score = bag_score.squeeze(0)
            patch_scores = patch_scores.squeeze(0)
        if return_attention:
            return bag_score, patch_scores, None
        return bag_score, patch_scores, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, use_mlp_head=True)
