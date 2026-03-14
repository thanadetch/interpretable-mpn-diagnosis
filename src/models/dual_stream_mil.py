import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        topk: int = 5,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
    ):
        """
        Robust Dual-Branch Attention MIL (Optimized for Stage-2: ET vs PV)
        Branch 1: Global Soft Attention (PV focus)
        Branch 2: Learnable Weighted Top-K Attention (ET focus)
        """
        super().__init__()
        self.topk = topk
        self.input_dim = input_dim

        # --- Branch 1: Global Expert ---
        self.global_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # --- Branch 2: Focal Expert ---
        self.focal_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # --- Robust Fusion Classifier (MLP + Regularization) ---
        self.fusion_classifier = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs):
        # Safe batch handling
        if x.dim() == 3:
            assert x.size(0) == 1, (
                "Model currently only supports batch_size=1 (bag-by-bag)."
            )
            x = x.squeeze(0)

        N, D = x.shape

        # ==========================================
        # Branch 1: Global Soft Attention
        # ==========================================
        A_global_raw = self.global_attention(x)  # [N, 1]
        A_global = F.softmax(A_global_raw, dim=0)  # [N, 1]
        global_feature = torch.mm(A_global.t(), x)  # [1, D]

        # ==========================================
        # Branch 2: Focal Weighted Top-K Attention
        # ==========================================
        A_focal_raw = self.focal_attention(x)  # [N, 1]
        A_focal_soft = F.softmax(A_focal_raw, dim=0)  # Convert to probabilities first

        k = min(self.topk, N)

        # Extract Top-K probabilities and their indices
        top_A_focal, focal_indices = torch.topk(A_focal_soft, k, dim=0)

        # Re-normalize ONLY the top-k scores so they sum to 1 (Weighted Mean)
        top_A_focal_norm = top_A_focal / (top_A_focal.sum() + 1e-8)

        # Multiply re-normalized scores by their respective patches
        top_features = x[focal_indices.squeeze(1)]  # [k, D]
        focal_feature = torch.mm(top_A_focal_norm.t(), top_features)  # [1, D]

        # ==========================================
        # Fusion & Classification
        # ==========================================
        # Concat: [1, D] + [1, D] -> [1, 2*D]
        fused_feature = torch.cat([global_feature, focal_feature], dim=1)

        # Pass through MLP with Dropout
        logits = self.fusion_classifier(fused_feature)  # [1, num_classes]
        logits = logits.squeeze(0)  # [num_classes]

        return logits, A_global, A_focal_raw
