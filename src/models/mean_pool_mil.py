"""
MeanPoolMIL: Standard Mean-Pooling baseline MIL (no attention mechanism).

Architecture:
    Mean Pooling: [N, vision_dim] → mean → [vision_dim]
    Classifier: Dropout → Linear(vision_dim, num_classes)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MeanPoolMIL(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vision_dim, num_classes),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # Handle both [B, N, D] and [N, D] shapes safely
        if vision_features.dim() == 3:
            bag_emb = vision_features.mean(dim=1)
        else:
            bag_emb = vision_features.mean(dim=0, keepdim=True)

        logits = self.classifier(bag_emb)

        if vision_features.dim() == 2 and logits.dim() == 2:
            logits = logits.squeeze(0)

        return logits, None, None
