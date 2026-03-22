"""
MultiTaskMIL: Gated-Attention MIL with auxiliary cellularity head.

Architecture:
    Bottleneck: Linear(input_dim, hidden_dim) -> ReLU -> Dropout
    Gated Attention: tanh/sigmoid gating -> softmax attention
    Two heads from the pooled bag embedding:
        - subtype_head:     Linear(hidden_dim, num_classes)  (ET vs PV)
        - cellularity_head: Linear(hidden_dim, 3)            (Hypo / Normo / Hyper)

The forward method returns (subtype_logits, cellularity_logits, attention_weights).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiTaskMIL(nn.Module):
    """
    Multi-task Gated-Attention MIL with auxiliary cellularity supervision.

    Returns (subtype_logits, cellularity_logits, attention_weights)
    to support joint training with a primary subtype loss and an
    auxiliary cellularity loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
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

        # Primary head: subtype classification (ET vs PV)
        self.subtype_head = nn.Linear(hidden_dim, num_classes)

        # Auxiliary head: cellularity classification (3 classes)
        self.cellularity_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: Instance features [N, D].
            return_attention: Kept for API compatibility (attention is always returned).
            metrics: Unused, kept for API compatibility.

        Returns:
            subtype_logits:     Bag-level subtype logits [num_classes].
            cellularity_logits: Bag-level cellularity logits [3].
            attention:          Attention weights [N].
        """
        # Bottleneck
        h = self.bottleneck(features)  # [N, hidden_dim]

        # Gated attention scores
        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        # Attention-weighted aggregation
        bag_emb = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden_dim]

        # Two heads
        subtype_logits = self.subtype_head(bag_emb)  # [num_classes]
        cellularity_logits = self.cellularity_head(bag_emb)  # [3]

        return subtype_logits, cellularity_logits, attention
