"""
LogitPriorFusionMIL: Logit-space Prior Fusion MIL.

Architecture:
    Vision Branch:
        bottleneck → gated attention → bag_emb → vision_classifier → vision_logits
    Concept Branch:
        concept_classifier (tiny linear) → concept_logits
    Fusion:
        final_logits = vision_logits + alpha * concept_logits

The concept prior acts as a learned additive bias in logit space,
allowing the vision branch to operate normally while the concept
provides a soft prior. The learnable scalar `alpha` controls how
strongly the concept influences the final prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LogitPriorFusionMIL(nn.Module):
    """
    Logit-space Prior Fusion MIL for ET vs PV classification.

    Adds a learnable concept-based logit prior to the vision model's
    logits, preventing shortcut overfitting while still allowing the
    concept to influence predictions.

    Args:
        vision_dim: Dimensionality of patch-level vision features.
        concept_dim: Dimensionality of the concept vector (default: 2).
        num_classes: Number of output classes (default: 2).
        hidden_dim: Bottleneck dimension for the attention network (default: 128).
        dropout: Dropout rate in the bottleneck (default: 0.5).

    Returns (logits, attention_weights, None) to match existing MIL forward signatures.
    """

    def __init__(
        self,
        vision_dim: int,
        concept_dim: int = 2,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        # ── Vision Branch ────────────────────────────────────────────
        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Vision classifier
        self.vision_classifier = nn.Linear(hidden_dim, num_classes)

        # ── Concept Branch ───────────────────────────────────────────
        # Tiny linear: [is_hyper, confidence] → logits
        self.concept_classifier = nn.Linear(concept_dim, num_classes)

        # ── Fusion ───────────────────────────────────────────────────
        # Learnable scalar controlling concept influence
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        vision_features: torch.Tensor,
        concept_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            vision_features: Patch-level features [N, vision_dim].
            concept_features: Concept vector [1, concept_dim] or [concept_dim].

        Returns:
            logits: Bag-level logits [num_classes].
            attention: Attention weights [N].
            None: Placeholder for compatibility with other MIL signatures.
        """
        # ── Vision branch ─────────────────────────────────────────────
        h = self.bottleneck(vision_features)  # [N, hidden_dim]

        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        # Attention-weighted aggregation → bag embedding
        bag_emb = torch.mm(attention.unsqueeze(0), h)  # [1, hidden_dim]

        vision_logits = self.vision_classifier(bag_emb).squeeze(0)  # [num_classes]

        # ── Concept branch ────────────────────────────────────────────
        if concept_features.dim() == 1:
            concept_features = concept_features.unsqueeze(0)  # [1, concept_dim]
        concept_logits = self.concept_classifier(concept_features).squeeze(0)  # [num_classes]

        # ── Fusion: additive logit prior ──────────────────────────────
        final_logits = vision_logits + self.alpha * concept_logits

        return final_logits, attention, None
