"""
MeanTopKPoolMIL: Mean and Top-K Pooling MIL.

Captures both the global context (mean of all patches) and the most salient 
regional features (mean of the top-K patch values per feature dimension).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class MeanTopKPoolMIL(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,
        topk: int = 5,
    ) -> None:
        super().__init__()
        self.topk = topk
        # Concatenating Mean and Top-K Mean doubles the feature dimension
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(vision_dim * 2, num_classes),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # vision_features shape: [B, N, D] or [N, D]
        dim_to_reduce = 1 if vision_features.dim() == 3 else 0
        
        # Number of patches (N)
        N = vision_features.size(dim_to_reduce)
        # Handle cases where the bag has fewer patches than K
        actual_k = min(self.topk, N)
        
        # 1. Mean (Global context)
        feat_mean = vision_features.mean(dim=dim_to_reduce, keepdim=True)
        
        # 2. Top-K Mean (Most salient features per channel)
        topk_vals = vision_features.topk(k=actual_k, dim=dim_to_reduce)[0]
        feat_topk = topk_vals.mean(dim=dim_to_reduce, keepdim=True)
        
        # Concatenate: [..., 2 * D]
        bag_emb = torch.cat([feat_mean, feat_topk], dim=-1)
        
        logits = self.classifier(bag_emb)
        
        if vision_features.dim() == 2 and logits.dim() == 2:
            logits = logits.squeeze(0)

        return logits, None, None

