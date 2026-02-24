"""
VisionLanguage_MIL: Text-as-Classifier MIL for zero-shot text-prior matching.

Instead of a traditional nn.Linear classifier, this model computes cosine
similarity between the attention-pooled bag vector and frozen text
prototypes (one per class), scaled by a learnable temperature.

Architecture:
    Attention:  Linear(D, 128) -> Tanh -> Linear(128, 1) -> Softmax
    Bag vector: attention-weighted sum of patch features  [D]
    Classifier: L2-normalize bag vector, dot-product with text_prototypes,
                scale by exp(logit_scale)

Usage:
    model  = VisionLanguage_MIL(feature_dim=768)
    logits, attn = model(image_features, text_prototypes)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VisionLanguage_MIL(nn.Module):
    """
    Attention-based MIL with text-prototype classification.

    Args:
        feature_dim: Dimensionality of input patch features (e.g. 768 for TITAN).
        hidden_dim:  Hidden dimension in the attention network.
        dropout:     Dropout rate after the attention hidden layer.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # ── Attention network ─────────────────────────────────────────
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # ── Learnable temperature (logit scale) ──────────────────────
        # Initialized to ln(100) ≈ 4.6052, following CLIP convention
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(100.0)),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features:  Patch features  [N, D]  or  [B, N, D].
            text_prototypes: Class embeddings [C, D]  (frozen, L2-normalized).

        Returns:
            logits:    Classification logits        [C]  or  [B, C].
            attention: Attention weights over patches [N] or [B, N].
        """
        # Handle both unbatched [N, D] and batched [B, N, D] inputs
        unbatched = image_features.dim() == 2
        if unbatched:
            image_features = image_features.unsqueeze(0)  # [1, N, D]

        # 1. Attention scores  ──  [B, N, 1] -> [B, N]
        attn_scores = self.attention(image_features).squeeze(-1)  # [B, N]
        attention = F.softmax(attn_scores, dim=-1)  # [B, N]

        # 2. Bag vector via attention-weighted sum  ──  [B, D]
        bag_vector = torch.bmm(
            attention.unsqueeze(1),  # [B, 1, N]
            image_features,  # [B, N, D]
        ).squeeze(1)  # [B, D]

        # 3. L2-normalize the bag vector
        bag_vector = F.normalize(bag_vector, p=2, dim=-1)  # [B, D]

        # 4. Dot product with text prototypes  ──  [B, C]
        logits = bag_vector @ text_prototypes.t()  # [B, C]

        # 5. Scale by learnable temperature
        logits = logits * self.logit_scale.exp()

        # Restore original shape if unbatched
        if unbatched:
            logits = logits.squeeze(0)  # [C]
            attention = attention.squeeze(0)  # [N]

        return logits, attention
