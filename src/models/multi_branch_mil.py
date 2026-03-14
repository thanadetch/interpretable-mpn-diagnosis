import torch
import torch.nn as nn


class MultiBranchMIL(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 3,
        topk_focal: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.topk_focal = topk_focal
        self.input_dim = input_dim

        # --- Branch 1: Global Cellularity (PV) ---
        self.global_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # --- Branch 2: Focal Morphology (ET) - Differentiable Top-K ---
        self.focal_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # --- Branch 3: Context Branch (Baseline Pooling) ---
        # CRITICAL: Keep this layer to preserve the random weight initialization sequence.
        self.context_attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # --- Late Fusion Classifier ---
        self.classifier = nn.Linear(input_dim * 3, num_classes)

    def forward(self, x: torch.Tensor, **kwargs):
        if x.dim() == 3:
            x = x.squeeze(0)

        N, D = x.shape

        # ==========================================
        # 1. Global Branch (Soft Attention)
        # ==========================================
        A_global_raw = self.global_attention(x)
        A_global = torch.softmax(A_global_raw, dim=0)
        global_feature = torch.mm(A_global.t(), x)  # [1, D]

        # ==========================================
        # 2. Focal Branch (Differentiable Soft Top-K Pooling)
        # ==========================================
        A_focal_raw = self.focal_attention(x)  # [N, 1]
        k_focal = min(self.topk_focal, N)

        # 2.1 Get Top-K RAW scores and their indices
        topk_scores_raw, focal_indices = torch.topk(
            A_focal_raw, k_focal, dim=0
        )  # [k, 1]

        # 2.2 Apply Softmax ONLY on the Top-K scores (smooth weighting)
        topk_scores_soft = torch.softmax(topk_scores_raw, dim=0)  # [k, 1]

        # 2.3 Gather corresponding features
        topk_features = x[focal_indices.squeeze(1)]  # [k, D]

        # 2.4 Weighted sum (Differentiable operation!)
        focal_feature = torch.mm(topk_scores_soft.t(), topk_features)  # [1, D]

        # ==========================================
        # 3. Context Branch (All Patches Unweighted Pooling)
        # ==========================================
        A_context_raw = self.context_attention(x)
        _, context_indices = torch.topk(A_context_raw.squeeze(1), N)
        context_feature = x[context_indices].mean(dim=0, keepdim=True)  # [1, D]

        # ==========================================
        # Late Fusion & Classification
        # ==========================================
        fused_feature = torch.cat(
            [global_feature, focal_feature, context_feature], dim=1
        )  # [1, 3*D]

        logits = self.classifier(fused_feature)  # [1, num_classes]
        logits = logits.squeeze(0)  # [num_classes]

        return logits, A_global, A_focal_raw
