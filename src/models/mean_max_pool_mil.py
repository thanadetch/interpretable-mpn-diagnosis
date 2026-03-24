"""
MeanMaxPoolMIL: Mean + Max Pooling baseline MIL (no attention mechanism).

Architecture:
    Mean Pooling: [N, vision_dim] → mean(dim=0) → [vision_dim]
    Max Pooling:  [N, vision_dim] → max(dim=0)  → [vision_dim]
    Concatenate:  [vision_dim * 2]
    Classifier:   Dropout → Linear(vision_dim * 2, num_classes)

Captures both global context (mean) and the most salient local
features (max) from the patch embeddings without any learned attention.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MeanMaxPoolMIL(nn.Module):
    """
    Mean + Max Pooling MIL baseline (no attention).

    Concatenates mean-pooled and max-pooled patch features,
    then classifies the result. Captures both average morphology
    and the most extreme/salient patch signal.

    Args:
        vision_dim: Dimensionality of patch-level vision features.
        num_classes: Number of output classes (default: 2).
        dropout: Dropout rate before the classifier (default: 0.5).

    Returns (logits, None, None) to match existing MIL forward signatures.
    """

    def __init__(
        self,
        vision_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 2, num_classes),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            vision_features: Patch-level features [N, vision_dim].

        Returns:
            logits: Bag-level logits [num_classes].
            None: No attention weights.
            None: Placeholder for compatibility.
        """
        # Mean pooling: [N, vision_dim] → [vision_dim]
        mean_emb = vision_features.mean(dim=0)

        # Max pooling: [N, vision_dim] → [vision_dim]
        max_emb = vision_features.max(dim=0)[0]

        # Concatenate: [vision_dim * 2]
        bag_emb = torch.cat([mean_emb, max_emb], dim=0)

        # Classify
        logits = self.classifier(bag_emb)  # [num_classes]

        return logits, None, None
