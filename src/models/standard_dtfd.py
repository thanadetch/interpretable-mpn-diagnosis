"""
Standard DTFD-MIL (CVPR 2022 Baseline) for ablation study.

Faithful reimplementation of:
    Zhang et al., "DTFD-MIL: Double-Tier Feature Distillation Multiple Instance
    Learning for Histopathology Whole Slide Image Classification", CVPR 2022.

Architecture Overview:
    Projection -> Tier-1 (Gated Attention + Instance Classifier) ->
    MaxMin Feature Distillation -> Tier-2 Gated Attention -> Bag Classifier.

Key Difference from Custom DTFD-lite:
    This version uses **MaxMin feature distillation** to select the top-k and
    bottom-k scoring instances from each pseudo-bag, intentionally DROPPING
    intermediate patches. The custom DTFD-lite preserves ALL patches via
    attention-based aggregation, which is hypothesised to be better for small
    ROI bags in MPN subtype differentiation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class GatedAttention(nn.Module):
    """Gated Attention mechanism for MIL aggregation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        V = self.attention_V(x)
        U = self.attention_U(x)
        attn_scores = self.attention_W(self.dropout(V * U)).squeeze(-1)
        attention = F.softmax(attn_scores, dim=-1)

        aggregated = torch.bmm(attention.unsqueeze(1), x).squeeze(1)

        if squeeze_output:
            aggregated = aggregated.squeeze(0)
            attention = attention.squeeze(0)

        if return_attention:
            return aggregated, attention
        return aggregated, None


class InstanceClassifier(nn.Module):
    """Tier-1 instance-level classifier for scoring patches."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
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
        return self.fc(x)


class StandardDTFDMIL(nn.Module):
    """
    Standard DTFD-MIL with MaxMin Feature Distillation (CVPR 2022).

    Unlike the custom DTFD-lite which preserves ALL patches via attention
    aggregation, this version intentionally drops intermediate-scoring patches
    through MaxMin selection. This serves as an ablation baseline to demonstrate
    the benefit of preserving all morphological features for small ROI bags.

    Architecture:
        1. Project raw features to lower dimension.
        2. Score all instances with the instance classifier.
        3. MaxMin distillation: select top-k and bottom-k instances per
           pseudo-bag to form distilled pseudo-bag feature sets.
        4. Tier-1: Gated Attention aggregates each distilled pseudo-bag.
        5. Tier-2: Gated Attention aggregates pseudo-bag representations.
        6. Bag classifier produces final logits.

    Args:
        input_dim: Dimension of input features from backbone.
        num_classes: Number of output classes (default: 2 for ET vs PV).
        proj_dim: Dimension after projection (default: 512).
        hidden_dim: Hidden dimension for attention (default: 128).
        dropout: Dropout probability (default: 0.25).
        num_pseudo_bags: Number of pseudo-bags to create (default: 3).
        distill_k: Number of top + bottom instances to keep per pseudo-bag
                   (default: 1, meaning top-1 and bottom-1 = 2 instances kept).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        proj_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        num_pseudo_bags: int = 3,
        distill_k: int = 1,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_pseudo_bags = num_pseudo_bags
        self.proj_dim = proj_dim
        self.distill_k = distill_k

        # Feature projection (shared)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Instance classifier — used for MaxMin scoring
        self.instance_classifier = InstanceClassifier(
            input_dim=proj_dim,
            hidden_dim=proj_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Tier 1: Gated Attention over distilled instances
        self.tier1_attention = GatedAttention(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Tier 2: Gated Attention over pseudo-bag representations
        self.tier2_attention = GatedAttention(
            input_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Bag classifier (final)
        self.bag_classifier = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── helpers ────────────────────────────────────────────────────────

    def _normalize_bag_input(self, features: torch.Tensor) -> torch.Tensor:
        """Safely handle [N, D] or [1, N, D] inputs."""
        if features.dim() == 3:
            assert features.size(0) == 1, "Only batch size 1 is supported."
            features = features.squeeze(0)
        assert features.dim() == 2, f"Expected 2D tensor, got {features.dim()}D"
        return features

    def _create_pseudo_bags(
        self,
        features: torch.Tensor,
        num_pseudo_bags: int,
    ) -> List[torch.Tensor]:
        """Deterministic sequential split into pseudo-bags (no shuffle)."""
        N = features.size(0)
        effective_k = max(min(num_pseudo_bags, N), 1)

        instances_per_bag = N // effective_k
        remainder = N % effective_k

        pseudo_bags = []
        start = 0
        for i in range(effective_k):
            extra = 1 if i < remainder else 0
            end = start + instances_per_bag + extra
            if end > start:
                pseudo_bags.append(features[start:end])
            start = end
        return pseudo_bags

    def _maxmin_distill(
        self,
        features: torch.Tensor,
        instance_logits: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        MaxMin Feature Distillation.

        Selects the top-k and bottom-k scoring instances from the pseudo-bag
        based on the maximum class score, intentionally dropping intermediate
        patches.

        Args:
            features: Pseudo-bag features [M, D].
            instance_logits: Instance-level logits [M, C].
            k: Number of top/bottom instances to keep.

        Returns:
            Distilled features [2k, D] (or fewer if M < 2k).
        """
        M = features.size(0)
        # Use max class score as saliency
        scores = instance_logits.max(dim=-1).values  # [M]

        # Clamp k so we don't exceed available instances
        effective_k = min(k, M // 2) if M >= 2 else M
        if effective_k < 1:
            effective_k = 1

        if M <= 2 * effective_k:
            # Not enough instances to split — keep all
            return features

        _, top_idx = torch.topk(scores, effective_k, largest=True)
        _, bot_idx = torch.topk(scores, effective_k, largest=False)

        # Combine and deduplicate indices
        selected_idx = torch.cat([top_idx, bot_idx]).unique()
        return features[selected_idx]

    # ── forward methods ───────────────────────────────────────────────

    def forward_training(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Training forward with MaxMin distillation.

        Returns:
            bag_logits: Final bag-level logits [C].
            pseudo_bag_logits: Tier-1 pseudo-bag logits [K, C].
            instance_logits_list: List of instance logits per pseudo-bag.
        """
        features = self._normalize_bag_input(features)
        projected = self.projection(features)

        # Split into pseudo-bags BEFORE distillation
        pseudo_bags = self._create_pseudo_bags(projected, self.num_pseudo_bags)

        pseudo_bag_representations = []
        pseudo_bag_logits = []
        instance_logits_list = []

        for pseudo_bag in pseudo_bags:
            # Score all instances in this pseudo-bag
            inst_logits = self.instance_classifier(pseudo_bag)  # [M, C]
            instance_logits_list.append(inst_logits)

            # MaxMin distillation — drop intermediate patches
            distilled = self._maxmin_distill(pseudo_bag, inst_logits, self.distill_k)

            # Tier-1 attention over distilled features only
            aggregated, _ = self.tier1_attention(distilled)
            pseudo_bag_representations.append(aggregated)

            pseudo_logits = self.bag_classifier(aggregated)
            pseudo_bag_logits.append(pseudo_logits)

        pseudo_bag_features = torch.stack(pseudo_bag_representations, dim=0)
        pseudo_bag_logits = torch.stack(pseudo_bag_logits, dim=0)

        # Tier 2: aggregate pseudo-bag representations
        tier2_aggregated, _ = self.tier2_attention(pseudo_bag_features)
        bag_logits = self.bag_classifier(tier2_aggregated)

        return bag_logits, pseudo_bag_logits, instance_logits_list

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        """
        Inference forward with MaxMin distillation (mirrors training path).

        Returns:
            logits: Bag-level logits [C] or [1, C].
            None: Placeholder for API compatibility.
            tier2_attn: Tier-2 attention weights (if return_attention).
        """
        squeeze_output = (features.dim() == 2)
        features = self._normalize_bag_input(features)
        projected = self.projection(features)

        pseudo_bags = self._create_pseudo_bags(projected, self.num_pseudo_bags)

        pseudo_bag_representations = []
        for pseudo_bag in pseudo_bags:
            inst_logits = self.instance_classifier(pseudo_bag)
            distilled = self._maxmin_distill(pseudo_bag, inst_logits, self.distill_k)
            aggregated, _ = self.tier1_attention(distilled)
            pseudo_bag_representations.append(aggregated)

        pseudo_bag_features = torch.stack(pseudo_bag_representations, dim=0)

        tier2_aggregated, tier2_attn = self.tier2_attention(
            pseudo_bag_features, return_attention
        )
        logits = self.bag_classifier(tier2_aggregated)

        if not squeeze_output:
            logits = logits.unsqueeze(0)
            if tier2_attn is not None:
                tier2_attn = tier2_attn.unsqueeze(0)

        return logits, None, tier2_attn
