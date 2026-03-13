"""
ResidualMetricMIL: Late Residual ROI-Metric Branch for MIL.

Main driver  : SimpleGatedMIL-style gated attention on instance features.
Assistant     : Tiny MLP on 7 ROI-level density/context metrics,
               zero-initialized final layer, fused via learnable scalar alpha.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResidualMetricMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # ── A. Main image branch (SimpleGatedMIL pattern) ────────────
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        self.classifier_img = nn.Linear(hidden_dim, num_classes)

        # ── B. Tiny metric branch ────────────────────────────────────
        self.metric_mlp = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )
        # Zero-initialize final layer
        nn.init.zeros_(self.metric_mlp[-1].weight)
        nn.init.zeros_(self.metric_mlp[-1].bias)

        self.alpha = nn.Parameter(torch.tensor(0.1))

    # -----------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        metrics: Optional[dict] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            features: Instance features [N, D].
            metrics:  Optional dict with keys "tissue_frac", "border_white_frac", "internal_white_frac".
            return_attention: unused, kept for interface compat.

        Returns:
            logits:            Fused bag-level logits [C].
            attention_weights: Softmax attention over instances [N].
            None:              Placeholder for DTFD compat.
        """
        device = features.device

        # ── Image branch ─────────────────────────────────────────────
        h = self.bottleneck(features)  # [N, hidden_dim]

        a_v = self.attention_V(h)  # [N, hidden_dim]
        a_u = self.attention_U(h)  # [N, hidden_dim]
        att_logits = self.attention_W(a_v * a_u).squeeze(-1)  # [N]
        attention_weights = F.softmax(att_logits, dim=0)  # [N]

        z_attn = (attention_weights.unsqueeze(-1) * h).sum(dim=0)  # [hidden_dim]
        logits_img = self.classifier_img(z_attn)  # [C]

        # ── Metric extraction ────────────────────────────────────────
        g = self._extract_metrics(features, metrics, device)

        # ── Late residual fusion ─────────────────────────────────────
        logits_metrics = self.metric_mlp(g)
        logits = logits_img + (self.alpha * logits_metrics)

        return logits, attention_weights, None

    # -----------------------------------------------------------------
    @staticmethod
    def _extract_metrics(
        features: torch.Tensor,
        metrics: Optional[dict],
        device: torch.device,
    ) -> torch.Tensor:
        """Return a [7] tensor of roughly [-1, 1]-scaled ROI metrics."""
        zeros = torch.zeros(7, device=device)

        if metrics is None or "tissue_frac" not in metrics:
            return zeros

        try:
            tissue_frac = metrics["tissue_frac"]
            if not isinstance(tissue_frac, torch.Tensor) or tissue_frac.numel() == 0:
                return zeros
            tissue_frac = tissue_frac.float().to(device)

            border_white_frac = metrics.get("border_white_frac")
            if not isinstance(border_white_frac, torch.Tensor) or border_white_frac.numel() == 0:
                border_white_frac = torch.zeros_like(tissue_frac)
            else:
                border_white_frac = border_white_frac.float().to(device)

            internal_white_frac = metrics.get("internal_white_frac")
            if not isinstance(internal_white_frac, torch.Tensor) or internal_white_frac.numel() == 0:
                internal_white_frac = torch.zeros_like(tissue_frac)
            else:
                internal_white_frac = internal_white_frac.float().to(device)

            mean_tissue = (tissue_frac.mean().view(1) - 0.5) * 2.0
            mean_border_white = (border_white_frac.mean().view(1) - 0.5) * 2.0
            mean_internal_white = (internal_white_frac.mean().view(1) - 0.5) * 2.0
            frac_low_tissue = ((tissue_frac < 0.2).float().mean().view(1) - 0.5) * 2.0
            frac_high_border = ((border_white_frac > 0.8).float().mean().view(1) - 0.5) * 2.0
            p75_tissue = (torch.quantile(tissue_frac, 0.75).view(1) - 0.5) * 2.0
            log_n = (
                torch.log1p(
                    torch.tensor(
                        [features.size(0)], dtype=torch.float32, device=device
                    )
                )
                / 4.0
            ) - 1.0

            g = torch.cat(
                [
                    mean_tissue,
                    mean_border_white,
                    mean_internal_white,
                    frac_low_tissue,
                    frac_high_border,
                    p75_tissue,
                    log_n,
                ]
            )
            return g

        except Exception:
            return zeros
