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
            metrics:  Optional dict with keys "tissue", "bg", "space".
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

        if metrics is None or "tissue" not in metrics:
            return zeros

        try:
            tissue = metrics["tissue"]
            if not isinstance(tissue, torch.Tensor) or tissue.numel() == 0:
                return zeros
            tissue = tissue.float().to(device)

            bg = metrics.get("bg")
            if not isinstance(bg, torch.Tensor) or bg.numel() == 0:
                bg = torch.zeros_like(tissue)
            else:
                bg = bg.float().to(device)

            space = metrics.get("space")
            if not isinstance(space, torch.Tensor) or space.numel() == 0:
                space = torch.zeros_like(tissue)
            else:
                space = space.float().to(device)

            mean_tissue = (tissue.mean().view(1) - 0.5) * 2.0
            mean_bg = (bg.mean().view(1) - 0.5) * 2.0
            mean_space = (space.mean().view(1) - 0.5) * 2.0
            frac_low_tissue = ((tissue < 0.2).float().mean().view(1) - 0.5) * 2.0
            frac_high_bg = ((bg > 0.8).float().mean().view(1) - 0.5) * 2.0
            p75_tissue = (torch.quantile(tissue, 0.75).view(1) - 0.5) * 2.0
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
                    mean_bg,
                    mean_space,
                    frac_low_tissue,
                    frac_high_bg,
                    p75_tissue,
                    log_n,
                ]
            )
            return g

        except Exception:
            return zeros
