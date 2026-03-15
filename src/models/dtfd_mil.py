"""
DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning.

Implementation based on:
    Zhang et al., "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
    Learning for Histopathology Whole Slide Image Classification", CVPR 2022.

Architecture Overview:
    Tier 1 (Instance-Level): Operates on individual patches to generate attention
    scores and pseudo-labels. Consists of instance classifier and attention module.

    Tier 2 (Bag-Level): Uses attention-based aggregation (Gated Attention) to
    combine distilled features from pseudo-bags for final bag-level prediction.

Key Features:
    - Pseudo-bag generation during training to increase effective batch size
    - Two-tier feature distillation for better instance-level feature learning
    - Gated attention mechanism for weighted aggregation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GatedAttention(nn.Module):
    """
    Gated Attention mechanism for MIL aggregation.

    Uses a gating mechanism to control information flow:
        a = sigmoid(W_gate @ h) * tanh(W_attn @ h)
        alpha = softmax(a)

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Hidden dimension for attention computation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.attention_W = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Instance features [B, N, D] or [N, D].
            return_attention: Whether to return attention weights.

        Returns:
            aggregated: Aggregated bag representation [B, D] or [D].
            attention: Attention weights [B, N] or [N] (if return_attention=True).
        """
        # Handle both batched and unbatched input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Gated attention scores
        V = self.attention_V(x)  # [B, N, hidden_dim]
        U = self.attention_U(x)  # [B, N, hidden_dim]

        attn_scores = self.attention_W(self.dropout(V * U))  # [B, N, 1]
        attn_scores = attn_scores.squeeze(-1)  # [B, N]

        # Softmax to get attention weights
        attention = F.softmax(attn_scores, dim=-1)  # [B, N]

        # Weighted aggregation
        aggregated = torch.bmm(attention.unsqueeze(1), x)  # [B, 1, D]
        aggregated = aggregated.squeeze(1)  # [B, D]

        if squeeze_output:
            aggregated = aggregated.squeeze(0)
            attention = attention.squeeze(0)

        if return_attention:
            return aggregated, attention
        return aggregated, None


class InstanceClassifier(nn.Module):
    """
    Tier 1 Instance-level classifier.

    Generates pseudo-labels for individual instances (patches).

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Instance features [N, D] or [B, N, D].

        Returns:
            logits: Instance-level logits [N, C] or [B, N, C].
        """
        return self.fc(x)


class DTFDMIL(nn.Module):
    """
    Double-Tier Feature Distillation MIL model.

    The model consists of two tiers:
        - Tier 1: Instance-level classifier + Attention for pseudo-bag aggregation
        - Tier 2: Bag-level classifier using distilled features from pseudo-bags

    During training:
        1. Split the original bag into K pseudo-bags
        2. Tier 1 generates attention-weighted representations for each pseudo-bag
        3. Tier 2 aggregates pseudo-bag representations for final prediction

    During inference:
        The full bag is processed through both tiers.

    Args:
        input_dim: Dimension of input features from backbone (e.g., 768 for TITAN).
        num_classes: Number of output classes (default: 3 for ET/PV/PMF).
        proj_dim: Dimension after projection layer (default: 512).
        hidden_dim: Hidden dimension for attention (default: 128).
        dropout: Dropout probability (default: 0.25).
        num_pseudo_bags: Number of pseudo-bags to create during training (default: 8).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        proj_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        num_pseudo_bags: int = 8,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_pseudo_bags = num_pseudo_bags
        self.proj_dim = proj_dim

        # Feature projection layer (shared)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Tier 1: Instance-level components
        self.instance_classifier = InstanceClassifier(
            input_dim=proj_dim,
            hidden_dim=proj_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.tier1_attention = GatedAttention(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Tier 2: Bag-level classifier
        # Takes distilled features from pseudo-bags
        self.tier2_attention = GatedAttention(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.bag_classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Learnable temperature for VL Cosine Similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def _create_pseudo_bags(
        self,
        features: torch.Tensor,
        num_pseudo_bags: int,
    ) -> List[torch.Tensor]:
        """
        Split a bag of instances into multiple pseudo-bags.

        Uses random shuffling and even splitting to create diverse pseudo-bags.

        Args:
            features: Instance features [N, D].
            num_pseudo_bags: Number of pseudo-bags to create.

        Returns:
            List of pseudo-bag tensors, each [N_k, D].
        """
        N = features.size(0)

        # CRITICAL FIX: Deterministic splitting during inference
        if self.training:
            indices = torch.randperm(N, device=features.device)
        else:
            indices = torch.arange(N, device=features.device)

        shuffled = features[indices]

        # Calculate instances per pseudo-bag
        instances_per_bag = N // num_pseudo_bags
        remainder = N % num_pseudo_bags

        pseudo_bags = []
        start = 0
        for i in range(num_pseudo_bags):
            # Distribute remainder instances
            extra = 1 if i < remainder else 0
            end = start + instances_per_bag + extra

            if end > start:  # Only add non-empty bags
                pseudo_bags.append(shuffled[start:end])
            start = end

        return pseudo_bags

    def forward_tier1(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Tier 1 forward pass.

        Args:
            features: Projected features [N, D] or [B, N, D].
            return_attention: Whether to return attention weights.

        Returns:
            instance_logits: Instance-level predictions [N, C] or [B, N, C].
            aggregated: Attention-weighted bag representation [D] or [B, D].
            attention: Attention weights (if return_attention=True).
        """
        instance_logits = self.instance_classifier(features)
        aggregated, attention = self.tier1_attention(features, return_attention)
        return instance_logits, aggregated, attention

    def forward_tier2(
        self,
        pseudo_bag_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Tier 2 forward pass.

        Args:
            pseudo_bag_features: Stacked pseudo-bag representations [K, D] or [B, K, D].
            return_attention: Whether to return attention weights.

        Returns:
            bag_logits: Bag-level predictions [C] or [B, C].
            attention: Attention weights (if return_attention=True).
        """
        aggregated, attention = self.tier2_attention(
            pseudo_bag_features, return_attention
        )
        bag_logits = self.bag_classifier(aggregated)
        return bag_logits, attention

    def forward_training(
        self,
        features: torch.Tensor,
        text_embeddings: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass during training with pseudo-bag creation.

        Args:
            features: Raw instance features [N, D] (single bag, no batch dim).
            text_embeddings: Optional VL text prototypes [num_classes, input_dim].

        Returns:
            bag_logits: Final bag-level logits [C].
            pseudo_bag_logits: Tier 1 logits for each pseudo-bag [K, C].
            instance_logits_list: List of instance logits per pseudo-bag.
            logits_t: Text-guided logits [C] (None if no text_embeddings).
        """
        # Project features
        projected = self.projection(features)  # [N, proj_dim]

        # Create pseudo-bags
        pseudo_bags = self._create_pseudo_bags(projected, self.num_pseudo_bags)

        # Tier 1: Process each pseudo-bag
        pseudo_bag_representations = []
        pseudo_bag_logits = []
        instance_logits_list = []

        for pseudo_bag in pseudo_bags:
            inst_logits, aggregated, _ = self.forward_tier1(pseudo_bag)
            instance_logits_list.append(inst_logits)
            pseudo_bag_representations.append(aggregated)

            # Pseudo-bag level prediction (for auxiliary loss)
            pseudo_logits = self.bag_classifier(aggregated)
            pseudo_bag_logits.append(pseudo_logits)

        # Stack pseudo-bag representations [K, D]
        pseudo_bag_features = torch.stack(pseudo_bag_representations, dim=0)
        pseudo_bag_logits = torch.stack(pseudo_bag_logits, dim=0)  # [K, C]

        # Tier 2: Aggregate pseudo-bags explicitly to get the aggregated feature
        tier2_aggregated, _ = self.tier2_attention(pseudo_bag_features)
        bag_logits = self.bag_classifier(tier2_aggregated)

        # Compute Text Logits safely
        logits_t = None
        if text_embeddings is not None:
            assert text_embeddings.size(1) == self.input_dim
            # CRITICAL FIX: Use self.projection[0] (only Linear) to avoid ReLU/Dropout on text
            text_proj = self.projection[0](text_embeddings)

            image_feat_norm = F.normalize(tier2_aggregated.squeeze(), dim=-1)
            text_emb_norm = F.normalize(text_proj, dim=-1)
            logit_scale = self.logit_scale.exp()
            logits_t = logit_scale * torch.matmul(image_feat_norm, text_emb_norm.t())

        return bag_logits, pseudo_bag_logits, instance_logits_list, logits_t

    def forward(
        self,
        features: torch.Tensor,
        text_embeddings: torch.Tensor = None,
        return_attention: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Standard forward pass (for inference).

        Uses the same pseudo-bag + Tier-2 aggregation path as forward_training()
        to avoid train-test feature distribution mismatch.

        Args:
            features: Instance features [B, N, D] or [N, D].
            text_embeddings: Optional VL text prototypes [num_classes, input_dim].
            return_attention: Whether to return attention weights.

        Returns:
            logits_v: Visual bag-level predictions [B, C] or [C].
            logits_t: Text-guided logits [B, C] or [C] (None if no text_embeddings).
            tier1_attention: Always None (not available with pseudo-bag path).
            tier2_attention: Tier 2 attention weights (if return_attention).
        """
        squeeze_output = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze_output = True

        B, N, D = features.shape
        assert B == 1, "Inference currently supports batch size 1 for pseudo-bag logic"

        # 1. Project features
        projected = self.projection(features.squeeze(0))  # [N, proj_dim]

        # 2. Create pseudo-bags (Deterministic during eval)
        pseudo_bags = self._create_pseudo_bags(projected, self.num_pseudo_bags)

        # 3. Tier 1 Aggregation
        pseudo_bag_representations = []
        for pseudo_bag in pseudo_bags:
            _, aggregated, _ = self.forward_tier1(pseudo_bag)
            pseudo_bag_representations.append(aggregated)

        pseudo_bag_features = torch.stack(pseudo_bag_representations, dim=0)  # [K, proj_dim]

        # 4. Tier 2 Aggregation (Visual Head)
        tier2_aggregated, tier2_attn = self.tier2_attention(
            pseudo_bag_features, return_attention=return_attention
        )
        logits_v = self.bag_classifier(tier2_aggregated)

        # 5. VL Head (Matching Train Path precisely)
        logits_t = None
        if text_embeddings is not None:
            text_proj = self.projection[0](text_embeddings)
            image_feat_norm = F.normalize(tier2_aggregated.squeeze(), dim=-1)
            text_emb_norm = F.normalize(text_proj, dim=-1)
            logit_scale = self.logit_scale.exp()
            logits_t = logit_scale * torch.matmul(image_feat_norm, text_emb_norm.t())

        if not squeeze_output:
            logits_v = logits_v.unsqueeze(0)
            if logits_t is not None:
                logits_t = logits_t.unsqueeze(0)
            if tier2_attn is not None:
                tier2_attn = tier2_attn.unsqueeze(0)

        # Return tier2_attn as the final attention output
        return logits_v, logits_t, None, tier2_attn


def compute_dtfd_loss(
    bag_logits: torch.Tensor,
    pseudo_bag_logits: torch.Tensor,
    instance_logits_list: List[torch.Tensor],
    bag_label: int,
    criterion: nn.Module,
    tier1_weight: float = 0.5,
    instance_weight: float = 0.2,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the combined DTFD-MIL loss.

    The total loss consists of:
        1. Bag-level loss (Tier 2 output vs ground truth)
        2. Pseudo-bag loss (Tier 1 pseudo-bag predictions vs ground truth)
        3. Instance-level loss (optional, using bag label as pseudo-label)

    Args:
        bag_logits: Final bag prediction [C].
        pseudo_bag_logits: Pseudo-bag predictions [K, C].
        instance_logits_list: List of instance logits per pseudo-bag.
        bag_label: Ground truth bag label (integer).
        criterion: Loss function (e.g., CrossEntropyLoss).
        tier1_weight: Weight for pseudo-bag loss.
        instance_weight: Weight for instance-level loss.

    Returns:
        total_loss: Combined loss scalar.
        loss_dict: Dictionary with individual loss components.
    """
    device = bag_logits.device
    label_tensor = torch.tensor([bag_label], device=device)

    # Bag-level loss (main supervision)
    bag_loss = criterion(bag_logits.unsqueeze(0), label_tensor)

    # Pseudo-bag loss (auxiliary)
    K = pseudo_bag_logits.size(0)
    pseudo_labels = label_tensor.expand(K)
    pseudo_bag_loss = criterion(pseudo_bag_logits, pseudo_labels)

    # Instance-level loss (weak supervision using bag label)
    instance_loss = torch.tensor(0.0, device=device)
    if instance_weight > 0 and len(instance_logits_list) > 0:
        total_instances = sum(logits.size(0) for logits in instance_logits_list)
        if total_instances > 0:
            all_instance_logits = torch.cat(instance_logits_list, dim=0)  # [N_total, C]
            instance_labels = label_tensor.expand(total_instances)
            instance_loss = criterion(all_instance_logits, instance_labels)

    # Combined loss
    total_loss = (
        bag_loss
        + tier1_weight * pseudo_bag_loss
        + instance_weight * instance_loss
    )

    loss_dict = {
        "bag_loss": bag_loss.item(),
        "pseudo_bag_loss": pseudo_bag_loss.item(),
        "instance_loss": instance_loss.item() if isinstance(instance_loss, torch.Tensor) else instance_loss,
        "total_loss": total_loss.item(),
    }

    return total_loss, loss_dict


