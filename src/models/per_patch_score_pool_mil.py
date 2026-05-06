"""
PerPatchScorePoolingMIL: per-patch score pooling MIL aggregator for scalar
fibrosis grading.

Pipeline:
    patch features -> patch scalar score (h(f_i)) -> score aggregation -> ROI score

Aggregation modes:
    - "mean":                 mean of all patch scores (sanity baseline)
    - "quantile":             q-th quantile of patch scores (severity-focused)
    - "topk_mean":            mean of top-k patch scores (worst-region evidence)
    - "mean_quantile_hybrid": (1 - λ) · mean + λ · quantile  (recommended)

Patch scores are *latent* ROI-level evidence — there is no patch-level
supervision. They are returned as the "attention" slot of the standard MIL
forward signature so the existing heatmap / explain pipeline can visualise
them directly.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


VALID_MODES = ("mean", "quantile", "topk_mean", "mean_quantile_hybrid")


class PerPatchScorePoolingMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        mode: str = "mean_quantile_hybrid",
        quantile: float = 0.75,
        topk_ratio: float = 0.20,
        max_lambda: float = 0.50,
        init_lambda: float = 0.20,
        use_mlp_head: bool = True,
    ) -> None:
        super().__init__()

        assert num_classes == 1, (
            "PerPatchScorePoolingMIL is intended for scalar regression "
            "(num_classes=1). Use --formulation regression."
        )
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}, got {mode!r}"
        assert 0.0 < quantile < 1.0
        assert 0.0 < topk_ratio <= 1.0

        self.mode = mode
        self.quantile = quantile
        self.topk_ratio = topk_ratio
        self.max_lambda = max_lambda

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

        if self.mode == "mean_quantile_hybrid":
            init_value = init_lambda / max_lambda
            init_value = min(max(init_value, 1e-4), 1.0 - 1e-4)
            init_logit = torch.logit(torch.tensor(init_value, dtype=torch.float32))
            self.lambda_logit = nn.Parameter(init_logit)

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            features: [N, D] (single bag) or [B, N, D] (batched bags).

        Returns:
            bag_score:    [B, 1] (or [1] when input is [N, D]).
            patch_scores: [B, N] latent severity scores (acts as "attention").
            None:         placeholder for interface compatibility.
        """
        squeeze_back = False
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, N, D]
            squeeze_back = True

        _, N, _ = features.shape

        patch_scores = self.score_head(features).squeeze(-1)  # [B, N]

        if self.mode == "mean":
            bag_score = patch_scores.mean(dim=1, keepdim=True)

        elif self.mode == "quantile":
            bag_score = torch.quantile(
                patch_scores, q=self.quantile, dim=1, keepdim=True
            )

        elif self.mode == "topk_mean":
            k = max(1, int(round(N * self.topk_ratio)))
            k = min(k, N)
            topk_scores, _ = torch.topk(patch_scores, k=k, dim=1)
            bag_score = topk_scores.mean(dim=1, keepdim=True)

        else:  # "mean_quantile_hybrid"
            mean_score = patch_scores.mean(dim=1, keepdim=True)
            q_score = torch.quantile(
                patch_scores, q=self.quantile, dim=1, keepdim=True
            )
            lam = self.max_lambda * torch.sigmoid(self.lambda_logit)
            bag_score = (1.0 - lam) * mean_score + lam * q_score

        if squeeze_back:
            bag_score = bag_score.squeeze(0)
            patch_scores = patch_scores.squeeze(0)

        return bag_score, patch_scores, None

    def get_lambda(self) -> Optional[float]:
        if not hasattr(self, "lambda_logit"):
            return None
        with torch.no_grad():
            lam = self.max_lambda * torch.sigmoid(self.lambda_logit)
        return float(lam.item())


