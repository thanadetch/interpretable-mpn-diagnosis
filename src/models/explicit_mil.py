"""
ExplicitMetricsMIL: Gated-Attention MIL with explicit global-metrics side-channel.

Architecture:
    Bottleneck: Linear(input_dim, hidden_dim) -> ReLU -> Dropout
    Stream A (Local): Gated-Attention pooling -> z_attn  [hidden_dim]
    Stream B (Global): 8 hand-crafted scalar metrics -> g  [8]
    Fusion: concat([z_attn, g]) -> Linear(hidden_dim + 8, num_classes)

The 8 explicit metrics capture ROI-level density and composition without
noisy high-dimensional embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ExplicitMetricsMIL(nn.Module):
    """
    Gated-Attention MIL with an explicit 8-scalar global-metrics side-channel.

    Returns (logits, attention_weights, None) to match other MIL forward signatures.
    """

    NUM_METRICS = 8

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated attention (Stream A)
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Classifier: local attention pool + 8 explicit metrics
        self.classifier = nn.Linear(hidden_dim + self.NUM_METRICS, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            features: Instance features [N, D].
            return_attention: Whether to return attention weights.
            metrics: Optional dict with keys 'tissue', 'space', 'bg', 'bad'
                     (each a tensor of length N).

        Returns:
            logits: Bag-level logits [C].
            attention: Attention weights [N] or None.
            None: Placeholder for API compatibility.
        """
        # ── Bottleneck ────────────────────────────────────────────────
        h = self.bottleneck(features)  # [N, hidden_dim]

        # ── Stream A: Gated-Attention pooling ─────────────────────────
        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]
        z_attn = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden_dim]

        # ── Stream B: Explicit global metrics (8 scalars) ─────────────
        if metrics is not None and "space" in metrics:
            tissue = metrics["tissue"].to(h.device).float()
            space = metrics["space"].to(h.device).float()
            bg = metrics["bg"].to(h.device).float()
            bad = metrics["bad"].to(h.device).float()

            mean_tissue = tissue.mean().view(1)
            mean_space = space.mean().view(1)
            mean_bg = bg.mean().view(1)
            mean_bad = bad.mean().view(1)
            space_to_tissue = (space / (tissue + 1e-6)).mean().view(1)
            frac_bad = (bad > 0.5).float().mean().view(1)
            n_patches = torch.tensor(
                [features.size(0) / 100.0], dtype=torch.float32, device=h.device
            )
            tissue_frac = (tissue > 0.15).float().mean().view(1)

            g = torch.cat(
                [
                    mean_tissue,
                    mean_space,
                    mean_bg,
                    mean_bad,
                    space_to_tissue,
                    frac_bad,
                    n_patches,
                    tissue_frac,
                ],
                dim=0,
            )  # [8]
        else:
            g = torch.zeros(self.NUM_METRICS, device=h.device)

        # ── Fusion & classifier ───────────────────────────────────────
        z_combined = torch.cat([z_attn, g], dim=0)  # [hidden_dim + 8]
        logits = self.classifier(z_combined)  # [num_classes]

        return logits, attention if return_attention else None, None
