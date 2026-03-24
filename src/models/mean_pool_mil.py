"""
MeanPoolMIL: Mean-Pooling baseline MIL (no attention mechanism).

Architecture:
    Mean Pooling: [N, vision_dim] → mean(dim=0) → [vision_dim]
    Classifier: Dropout → Linear(vision_dim, num_classes)

This is a simple baseline to verify whether Gated-Attention
introduces overfitting on small datasets (~40 patches per bag).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MeanPoolMIL(nn.Module):
    """
    Mean-Pooling MIL baseline (no attention).

    Simply mean-pools all patch features and classifies the result.
    Useful as a baseline to isolate whether attention-based aggregation
    is a source of overfitting on small datasets.

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
            nn.Linear(vision_dim, num_classes),
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
            None: No attention weights (mean pooling).
            None: Placeholder for compatibility.
        """
        # Mean pooling: [N, vision_dim] → [vision_dim]
        bag_emb = vision_features.mean(dim=0)  # [vision_dim]

        # Classify
        logits = self.classifier(bag_emb)  # [num_classes]

        return logits, None, None
