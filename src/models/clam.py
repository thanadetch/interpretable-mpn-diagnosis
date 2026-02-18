"""
CLAM-SB: Clustering-constrained Attention Multiple Instance Learning (Single Branch).

Implementation based on:
    Lu et al., "Data-efficient and weakly supervised computational pathology
    on whole-slide images", Nature Biomedical Engineering, 2021.

Architecture Overview:
    - Feature projection bottleneck (input_dim → 512)
    - Single-branch gated attention for bag-level aggregation
    - Bag classifier on attention-weighted representation
    - Instance-level clustering constraint: per-class binary classifiers
      trained on the top-k / bottom-k attended patches

Key Features:
    - Gated attention mechanism with single attention branch (SB variant)
    - Instance clustering loss acts as a regulariser, encouraging the
      attention module to identify class-discriminative patches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CLAM_SB(nn.Module):
    """
    CLAM Single-Branch model for MIL classification.

    Args:
        input_dim: Dimension of input features from backbone (e.g. 768).
        num_classes: Number of output classes (default: 3 for ET/PV/PMF).
        proj_dim: Hidden dimension after projection (default: 512).
        dropout: Dropout probability (default: 0.25).
        inst_k: Number of top/bottom patches for instance clustering
                 constraint. If 0, uses min(8, N//4) dynamically (default: 0).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        proj_dim: int = 512,
        dropout: float = 0.25,
        inst_k: int = 0,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.inst_k = inst_k

        # ── Feature projection ───────────────────────────────────────
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Gated attention (single branch) ──────────────────────────
        self.attention_V = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(128, 1)

        # ── Bag-level classifier ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # ── Instance-level classifiers (one binary head per class) ───
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(proj_dim, 2) for _ in range(num_classes)]
        )

        # Loss for instance clustering (internal)
        self.inst_criterion = nn.CrossEntropyLoss()

    # ── helpers ───────────────────────────────────────────────────────
    def _get_k(self, n_instances: int) -> int:
        """Determine k for instance clustering constraint."""
        if self.inst_k > 0:
            return min(self.inst_k, n_instances)
        return min(8, max(1, n_instances // 4))

    def _compute_attention(
        self,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gated attention scores.

        Args:
            h: Projected features [N, proj_dim].

        Returns:
            attention: Normalised attention weights [N].
            attn_raw: Raw attention scores [N] (before softmax).
        """
        V = self.attention_V(h)  # [N, 128]
        U = self.attention_U(h)  # [N, 128]
        attn_raw = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_raw, dim=0)  # [N]
        return attention, attn_raw

    def _inst_clustering_loss(
        self,
        h: torch.Tensor,
        attention: torch.Tensor,
        bag_label: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute instance-level clustering loss.

        For the *predicted/true* class:
            - top-k attended patches → positive (label 1)
            - bottom-k attended patches → negative (label 0)
        For every *other* class:
            - top-k attended patches → negative (label 0)

        Args:
            h: Projected features [N, proj_dim].
            attention: Attention weights [N].
            bag_label: Ground truth bag label (int).

        Returns:
            inst_loss: Scalar instance clustering loss.
            loss_dict: Breakdown of per-class instance losses.
        """
        device = h.device
        N = h.size(0)
        k = self._get_k(N)

        # Indices of highest and lowest attention patches
        _, top_idx = torch.topk(attention, k)
        _, bot_idx = torch.topk(attention, k, largest=False)

        total_loss = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, float] = {}

        for c in range(self.num_classes):
            inst_clf = self.instance_classifiers[c]

            if c == bag_label:
                # Positive class: top-k → 1 (positive), bottom-k → 0 (negative)
                top_logits = inst_clf(h[top_idx])  # [k, 2]
                bot_logits = inst_clf(h[bot_idx])  # [k, 2]

                top_labels = torch.ones(k, dtype=torch.long, device=device)
                bot_labels = torch.zeros(k, dtype=torch.long, device=device)

                logits = torch.cat([top_logits, bot_logits], dim=0)
                labels = torch.cat([top_labels, bot_labels], dim=0)
            else:
                # Other classes: top-k most attended → 0 (negative for this class)
                top_logits = inst_clf(h[top_idx])  # [k, 2]
                labels = torch.zeros(k, dtype=torch.long, device=device)
                logits = top_logits

            c_loss = self.inst_criterion(logits, labels)
            total_loss = total_loss + c_loss
            loss_dict[f"inst_c{c}"] = c_loss.item()

        # Average across classes
        total_loss = total_loss / self.num_classes
        return total_loss, loss_dict

    # ── forward (inference) ──────────────────────────────────────────
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Standard forward pass (inference).

        Args:
            features: Instance features [N, D].
            return_attention: Whether to return attention weights.

        Returns:
            logits: Bag-level logits [C].
            attention: Attention weights [N] (if return_attention=True).
            None: Placeholder for DTFD compatibility.
        """
        h = self.projection(features)  # [N, proj_dim]
        attention, _ = self._compute_attention(h)  # [N]

        # Attention-weighted aggregation
        aggregated = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [proj_dim]
        logits = self.classifier(aggregated)  # [C]

        if return_attention:
            return logits, attention, None
        return logits, None, None

    # ── forward_training ─────────────────────────────────────────────
    def forward_training(
        self,
        features: torch.Tensor,
        bag_label: int,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Training forward pass with instance clustering loss.

        Args:
            features: Instance features [N, D].
            bag_label: Ground truth bag label (int).

        Returns:
            logits: Bag-level logits [C].
            inst_dict: Dictionary containing:
                - inst_loss: Scalar instance clustering loss.
                - per-class instance loss breakdown.
        """
        h = self.projection(features)  # [N, proj_dim]
        attention, _ = self._compute_attention(h)  # [N]

        # Bag-level prediction
        aggregated = torch.mm(attention.unsqueeze(0), h).squeeze(0)
        logits = self.classifier(aggregated)  # [C]

        # Instance clustering loss
        inst_loss, inst_loss_dict = self._inst_clustering_loss(
            h,
            attention,
            bag_label,
        )

        inst_dict = {"inst_loss": inst_loss, **inst_loss_dict}
        return logits, inst_dict
