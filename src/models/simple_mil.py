"""
SimpleGatedMIL: Lightweight Gated-Attention MIL for small datasets.

Architecture:
    Bottleneck: Linear(input_dim, 128) -> ReLU -> Dropout(0.5)
    Gated Attention: tanh/sigmoid gating -> softmax attention
    Classifier: Linear(128, num_classes)

Supports optional Top-k pooling: when topk > 0, selects the k
highest-attention patches and mean-pools instead of attention-weighted avg.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SimpleGatedMIL(nn.Module):
    """
    Lightweight Gated-Attention MIL for small datasets.

    Architecture:
        Bottleneck: Linear(input_dim, 128) -> ReLU -> Dropout(0.5)
        Gated Attention: tanh/sigmoid gating -> softmax attention
        Classifier: Linear(128, num_classes)

    Returns (logits, attention_weights, None) to match DTFD forward signature.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        topk: int = 0,
    ) -> None:
        super().__init__()
        self.topk = topk

        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated attention
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

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
            metrics: Optional dict of patch-level metrics (for logit bias).

        Returns:
            logits: Bag-level logits [C].
            attention: Attention weights [N] (if return_attention=True).
            None: Placeholder for compatibility with DTFD signature.
        """
        # Bottleneck
        h = self.bottleneck(features)  # [N, hidden_dim]

        # Gated attention scores
        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]

        # --- Attention Logit Bias (1.1) ---
        if metrics is not None and "bad" in metrics:
            bad = metrics["bad"].to(attn_scores.device).float()
            bad = torch.clamp(bad, 0.0, 1.0)
            p_good = torch.clamp(1.0 - bad, min=1e-6, max=1.0)
            attn_scores = attn_scores + torch.log(p_good)
        # ----------------------------------

        attention = F.softmax(attn_scores, dim=0)  # [N]

        # Aggregation
        if self.topk > 0 and features.size(0) > self.topk:
            # Top-k pooling: select k patches with highest attention, mean-pool
            _, topk_idx = torch.topk(attention, self.topk)
            aggregated = h[topk_idx].mean(dim=0)  # [hidden_dim]
        else:
            # Standard attention-weighted average
            aggregated = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden_dim]

        # Classification
        logits = self.classifier(aggregated)  # [C]

        if return_attention:
            return logits, attention, None
        return logits, None, None
