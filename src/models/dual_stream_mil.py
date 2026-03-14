import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamMIL(nn.Module):
    """
    PV-ET Bi-Evidence MIL Architecture
    Branch A (PV): Global Soft Attention (captures diffuse panmyelosis / hypercellularity)
    Branch B (ET): Masked Soft Top-K Attention (captures focal giant megakaryocytes)
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        topk: int = 5,
        hidden_dim: int = 128,
        et_temp: float = 0.5,
    ):
        super().__init__()
        self.topk_et = topk
        self.et_temp = et_temp

        # 1. Shared Adapter (Reduces dimensionality and regularizes latent space)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. PV-Global Branch Attention (Broad evidence)
        self.pv_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # 3. ET-Focal Branch Attention (Sharp evidence)
        self.et_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

        # 4. Asymmetric Interaction Fusion Head
        # Concat: [pv, et, |pv - et|, pv * et] -> 4 * hidden_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, **kwargs):
        # Safe batch handling
        if x.dim() == 3:
            assert x.size(0) == 1, (
                "Model currently only supports batch_size=1 (bag-by-bag)."
            )
            x = x.squeeze(0)

        N, _ = x.shape

        # Apply Shared Adapter: [N, D] -> [N, hidden_dim]
        h = self.adapter(x)

        # ==========================================
        # Branch A: PV-Global (Broad Soft Attention)
        # ==========================================
        A_pv_raw = self.pv_attention(h)  # [N, 1]
        A_pv = torch.softmax(A_pv_raw, dim=0)  # [N, 1]
        pv_feat = torch.mm(A_pv.t(), h)  # [1, hidden_dim]

        # ==========================================
        # Branch B: ET-Focal (Masked Soft Top-K)
        # ==========================================
        A_et_raw = self.et_attention(h)  # [N, 1]
        k = min(self.topk_et, N)

        # Find top-k scores and indices
        topk_vals, topk_idx = torch.topk(A_et_raw, k, dim=0)

        # Create -inf mask and scatter top-k values into it
        mask = torch.full_like(A_et_raw, float("-inf"))
        mask.scatter_(0, topk_idx, topk_vals)

        # Softmax over mask (only top-k elements get probability mass)
        A_et = torch.softmax(mask / self.et_temp, dim=0)  # [N, 1]
        et_feat = torch.mm(A_et.t(), h)  # [1, hidden_dim]

        # ==========================================
        # Interaction Fusion
        # ==========================================
        # Calculate interactions
        diff_feat = torch.abs(pv_feat - et_feat)
        prod_feat = pv_feat * et_feat

        # Concatenate all evidences: [1, hidden_dim * 4]
        fusion = torch.cat([pv_feat, et_feat, diff_feat, prod_feat], dim=1)

        # Final prediction
        logits = self.fusion_head(fusion)  # [1, num_classes]
        logits = logits.squeeze(0)  # [num_classes]

        # Return 3 values to maintain compatibility with existing train loops for Exp 1
        return logits, A_pv_raw, A_et_raw
