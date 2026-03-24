"""
MeanPoolLateFusionMIL: Logit-level Late Fusion with Mean Pooling.

Architecture:
    Vision Branch:
        Mean Pooling → Dropout → Linear → vision_logits
    Concept Branch:
        Hardcoded rule: Hypercellular → PV, Non-hyper → ET
    Fusion:
        final_logits = vision_logits + alpha * concept_logits

The `alpha` is a fixed scalar weight (not learnable), so the
fusion balance is set as a hyperparameter. This avoids gradient
collapse issues seen with learnable alpha on small datasets.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MeanPoolLateFusionMIL(nn.Module):
    """
    Logit-level Late Fusion: Mean Pool (vision) + Rule-based (concept).

    Combines a mean-pooling vision classifier with a hardcoded
    rule-based concept logit using a fixed alpha weight.

    Args:
        vision_dim: Dimensionality of patch-level vision features.
        num_classes: Number of output classes (default: 2).
        dropout: Dropout rate before the vision classifier (default: 0.5).
        alpha: Fixed weight for the concept logit (default: 0.5).

    Returns (logits, None, None) to match existing MIL forward signatures.
    """

    def __init__(
        self,
        vision_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = alpha  # fixed, not nn.Parameter

        # Vision branch: mean pool → classify
        self.vision_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vision_dim, num_classes),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        concept_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            vision_features: Patch-level features [N, vision_dim] or [B, N, vision_dim].
            concept_features: Concept vector [1, concept_dim] or [concept_dim].

        Returns:
            logits: Bag-level logits [num_classes].
            None: No attention weights (mean pooling).
            None: Placeholder for compatibility.
        """
        # ── Vision branch ─────────────────────────────────────────────
        # Handle both [N, D] and [B, N, D] shapes
        if vision_features.dim() == 3:
            bag_emb = vision_features.mean(dim=1)  # [B, D]
        else:
            bag_emb = vision_features.mean(dim=0)  # [D]

        img_logit = self.vision_classifier(bag_emb)  # [num_classes] or [B, num_classes]

        # ── Concept Branch (Hardcoded Rule-based Logit) ──────────────────────
        if concept_features.dim() == 1:
            concept_features = concept_features.unsqueeze(0)

        # Extract [is_hyper, conf]
        is_hyper = concept_features[:, 0]  # 1.0 (Hyper) or 0.0 (Non-hyper)
        conf = concept_features[:, 1]      # Confidence (0.0 to 1.0)

        # Convert is_hyper (0, 1) to sign (-1, 1)
        sign = (is_hyper * 2.0) - 1.0

        # Rule: Hypercellular pushes towards PV, Non-hyper pushes towards ET
        csv_logit_pv = sign * conf
        csv_logit_et = -sign * conf

        # Stack into [B, 2] assuming index 0 is ET and index 1 is PV
        csv_logit = torch.stack([csv_logit_et, csv_logit_pv], dim=-1)

        # Ensure dimensions match for addition
        if vision_features.dim() == 2 and csv_logit.dim() == 2:
            csv_logit = csv_logit.squeeze(0)

        # ── Late Fusion ──────────────────────────────────────
        final_logit = img_logit + (self.alpha * csv_logit)

        return final_logit, None, None
