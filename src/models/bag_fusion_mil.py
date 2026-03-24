"""
BagFusionMIL: Bag-Level Fusion (Intermediate Feature-level Fusion) MIL.

Architecture:
    1. Gated Attention Network: vision_features [N, D] → bottleneck [N, 128]
       → gated attention → bag_emb [1, 128]
    2. Concept Projector: concept_features [1, 4] → projected [1, 128]
    3. Fusion: concat(bag_emb, concept_emb) → [1, 256]
    4. Classifier: Linear(256, num_classes) → logits

The concept vector encodes expert-assessed cellularity as a one-hot
[is_hypo, is_normo, is_hyper] plus a normalised confidence weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BagFusionMIL(nn.Module):
    """
    Bag-Level Fusion MIL for ET vs PV classification.

    Fuses gated-attention bag embedding with a projected concept vector
    (expert cellularity assessment) before the final classifier.

    Args:
        vision_dim: Dimensionality of patch-level vision features.
        concept_dim: Dimensionality of the concept vector (default: 4).
        projected_concept_dim: Output dim of the concept projector (default: 128).
        num_classes: Number of output classes (default: 2).
        hidden_dim: Bottleneck dimension for the attention network (default: 128).
        dropout: Dropout rate in the bottleneck (default: 0.5).

    Returns (logits, attention_weights, None) to match existing MIL forward signatures.
    """

    def __init__(
        self,
        vision_dim: int,
        concept_dim: int = 2,
        projected_concept_dim: int = 128,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── Gated Attention Network ──────────────────────────────────
        # Bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated attention mechanism (tanh / sigmoid gating)
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # ── Concept Projector ────────────────────────────────────────
        self.concept_projector = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.ReLU(),
            nn.Linear(64, projected_concept_dim),
            nn.ReLU(),
        )

        # ── Classifier ──────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim + projected_concept_dim, num_classes)

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
        # (a) Gated Attention → bag embedding
        h = self.bottleneck(vision_features)  # [N, hidden_dim]

        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        # Attention-weighted aggregation → bag embedding [1, hidden_dim]
        bag_emb = torch.mm(attention.unsqueeze(0), h)  # [1, hidden_dim]

        # (b) Concept projection
        if concept_features.dim() == 1:
            concept_features = concept_features.unsqueeze(0)  # [1, concept_dim]
        concept_emb = self.concept_projector(concept_features)  # [1, projected_concept_dim]

        # (c) Concatenate
        fused_bag = torch.cat([bag_emb, concept_emb], dim=1)  # [1, hidden_dim + projected_concept_dim]

        # (d) Classify
        logits = self.classifier(fused_bag).squeeze(0)  # [num_classes]

        return logits, attention, None
