"""
MeanStdPoolMIL: Mean and Standard Deviation Pooling MIL.

Captures both the global context (mean) and the heterogeneity/variance (std) 
of the patches in the ROI, without being overly sensitive to outliers (max).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class MeanStdPoolMIL(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # Concatenating Mean and Std doubles the feature dimension
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
        
        # 1. Mean (Global context)
        feat_mean = vision_features.mean(dim=dim_to_reduce, keepdim=True)
        
        # 2. Std (Heterogeneity/Variance). Unbiased=False prevents NaNs for N=1.
        feat_std = vision_features.std(dim=dim_to_reduce, keepdim=True, unbiased=False)
        
        # Concatenate: [..., 2 * D]
        bag_emb = torch.cat([feat_mean, feat_std], dim=-1)
        
        logits = self.classifier(bag_emb)
        
        if vision_features.dim() == 2 and logits.dim() == 2:
            logits = logits.squeeze(0)

        return logits, None, None
