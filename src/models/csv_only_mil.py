"""
CSVOnlyMIL: Concept-only baseline (no vision features).

Architecture:
    concept_classifier: Linear(concept_dim, num_classes)

Establishes a concept-only baseline (Exp 2) to measure the
predictive power of the cellularity CSV features alone.
Vision features are accepted but ignored to maintain the
standard forward(vision_features, concept_features) interface.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CSVOnlyMIL(nn.Module):
    """
    Concept-only MIL baseline.

    Classifies solely from the concept vector (e.g., [is_hyper, confidence]),
    ignoring all vision features. Used to establish an upper bound on
    how much signal the CSV concepts carry.

    Args:
        concept_dim: Dimensionality of the concept vector (default: 2).
        num_classes: Number of output classes (default: 2).

    Returns (logits, None, None) to match existing MIL forward signatures.
    """

    def __init__(
        self,
        concept_dim: int = 2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.concept_classifier = nn.Linear(concept_dim, num_classes)

    def forward(
        self,
        vision_features: torch.Tensor,
        concept_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            vision_features: Patch-level features [N, vision_dim] (IGNORED).
            concept_features: Concept vector [1, concept_dim] or [concept_dim].

        Returns:
            logits: Bag-level logits [num_classes].
            None: No attention weights.
            None: Placeholder for compatibility.
        """
        if concept_features.dim() == 1:
            concept_features = concept_features.unsqueeze(0)  # [1, concept_dim]

        logits = self.concept_classifier(concept_features).squeeze(0)  # [num_classes]

        return logits, None, None
