"""
EarlyFusionMIL: Patch-Level Early Fusion MIL.

Architecture:
    1. Concept Projector: concept_features [1, 4] → expand to [N, 4]
       → project to [N, projected_concept_dim]
    2. Patch-level concatenation: [N, vision_dim + projected_concept_dim]
    3. Bottleneck: [N, hidden_dim]
    4. Gated Attention: attention weights [N] → bag_emb [1, hidden_dim]
    5. Classifier: Linear(hidden_dim, num_classes) → logits

By fusing concepts at the patch level (before the attention bottleneck),
the model is forced to integrate concept information through the same
attention mechanism, preventing classifier shortcut overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EarlyFusionMIL(nn.Module):
    """
    Patch-Level Early Fusion MIL for ET vs PV classification.

    Concatenates a projected concept vector to each patch embedding
    BEFORE the attention bottleneck, forcing concept information through
    the gated attention mechanism.

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

        # ── Concept Projector ────────────────────────────────────────
        self.concept_projector = nn.Sequential(
            nn.Linear(concept_dim, 64),
            nn.ReLU(),
            nn.Linear(64, projected_concept_dim),
            nn.ReLU(),
        )

        # ── Bottleneck (takes fused patch features) ──────────────────
        self.bottleneck = nn.Sequential(
            nn.Linear(vision_dim + projected_concept_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Gated Attention ──────────────────────────────────────────
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # ── Classifier ──────────────────────────────────────────────
        self.classifier = nn.Linear(hidden_dim, num_classes)

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
        N = vision_features.size(0)

        # (a) Expand concept to match number of patches
        if concept_features.dim() == 1:
            concept_features = concept_features.unsqueeze(0)  # [1, concept_dim]
        concept_expanded = concept_features.expand(N, -1)  # [N, concept_dim]

        # (b) Project concepts
        concept_emb = self.concept_projector(concept_expanded)  # [N, projected_concept_dim]

        # (c) Patch-level concatenation
        fused_patches = torch.cat([vision_features, concept_emb], dim=1)  # [N, vision_dim + projected_concept_dim]

        # (d) Bottleneck
        h = self.bottleneck(fused_patches)  # [N, hidden_dim]

        # (e) Gated Attention → bag embedding
        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        # Attention-weighted aggregation → bag embedding [1, hidden_dim]
        bag_emb = torch.mm(attention.unsqueeze(0), h)  # [1, hidden_dim]

        # (f) Classify
        logits = self.classifier(bag_emb).squeeze(0)  # [num_classes]

        return logits, attention, None
