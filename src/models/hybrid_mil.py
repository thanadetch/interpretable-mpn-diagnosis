"""
HybridMIL: Fusion-based MIL combining Top-k Pooling and Soft Attention Pooling.

Architecture:
    Input Projection: Linear(input_dim, hidden_dim) -> ReLU -> Dropout
    Shared Gated Attention: tanh/sigmoid gating -> raw attention scores

    Branch 1 (Focal): Top-k patches by attention -> Mean Pool
    Branch 2 (Context): Softmax attention-weighted average of all patches

    Fusion: Concatenate Branch 1 & 2 -> [2 * hidden_dim]
    Classifier: Linear(2 * hidden_dim, num_classes)

The dual-branch design captures both the most discriminative patches (top-k)
and the global context (soft attention), fusing them for the final prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HybridMIL(nn.Module):
    """
    Fusion MIL: Top-k Pooling (focal) + Soft Attention Pooling (context).

    Args:
        input_dim: Dimension of input features from backbone.
        num_classes: Number of output classes (default: 3).
        hidden_dim: Hidden dimension after projection (default: 64).
        dropout: Dropout probability (default: 0.5).
        topk: Number of top-attended patches for Branch 1 (default: 8).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        topk: int = 5,
    ) -> None:
        super().__init__()
        self.topk = topk

        # ── Input projection ─────────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Shared gated attention ───────────────────────────────────
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # ── Classifier (takes concatenation of both branches) ────────
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            features: Instance features [N, D].
            return_attention: Whether to return attention weights.

        Returns:
            logits: Bag-level logits [C].
            attention: Softmax attention weights [N] (if return_attention=True).
            None: Placeholder for compatibility with other MIL models.
        """
        # Project
        h = self.projection(features)  # [N, hidden_dim]

        # Shared gated attention scores
        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        # Branch 1 — Focal (Top-k pooling)
        k = min(self.topk, features.size(0))
        _, topk_idx = torch.topk(attention, k)
        focal = h[topk_idx].mean(dim=0)  # [hidden_dim]

        # Branch 2 — Context (Soft attention pooling)
        context = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden_dim]

        # Fusion & classification
        fused = torch.cat([focal, context], dim=0)  # [2 * hidden_dim]
        logits = self.classifier(fused)  # [C]

        if return_attention:
            return logits, attention, None
        return logits, None, None
