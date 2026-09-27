"""a79 — Multi-mechanism ENSEMBLE aggregator (variance-reduction angle).

Every single mechanism ties the baseline and the real failure mode is VARIANCE
(val<->test decoupling up to 0.33 on this 7-10-patient cohort), not signal. An
internal ensemble of diverse, grading-aligned readouts that share one
bottleneck and are AVERAGED should reduce that variance without adding much
capacity:

    h = bottleneck(features)                        # [N,128]
    y_att  = head_att( gated_attention_pool(h) )    # learned salience
    y_mean = head_mean( mean_pool(h) )              # diffuse central density
    y_cov  = head_cov( coverage(<h, w>) )           # fraction fibre-positive
    y = (y_att + y_mean + y_cov) / 3

All three are grading-aligned (no ||h|| weighting); averaging decorrelates their
per-fold errors. Ablation a80: mode='attn_only' -> just the gated-attention head
= the locked baseline, isolating 'does averaging diverse mechanisms reduce
variance / improve over attention alone?'. Raw logits (trainer rounds+clips).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        mode: str = "ensemble",  # 'ensemble' | 'attn_only'
    ) -> None:
        super().__init__()
        self.mode = mode
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.head_att = nn.Linear(hidden_dim, num_classes)
        # diffuse-density branches (only used in ensemble mode)
        self.head_mean = nn.Linear(hidden_dim, num_classes)
        self.w_cov = nn.Parameter(torch.randn(hidden_dim) / (hidden_dim ** 0.5))
        self.tau = nn.Parameter(torch.zeros(1))
        self.log_beta = nn.Parameter(torch.zeros(1))
        self.head_cov = nn.Linear(1, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)  # [N, hidden]
        attn = F.softmax(self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1), dim=0)
        h_att = torch.mm(attn.unsqueeze(0), h)        # [1, hidden]
        y_att = self.head_att(h_att)                  # [1, C]
        if self.mode == "attn_only":
            return (y_att, attn, None) if return_attention else (y_att, None, None)

        y_mean = self.head_mean(h.mean(dim=0, keepdim=True))           # [1, C]
        s = h @ self.w_cov                                            # [N]
        beta = F.softplus(self.log_beta) + 1e-3
        cov = torch.sigmoid((s - self.tau) / beta).mean().view(1, 1)  # [1,1]
        y_cov = self.head_cov(cov)                                    # [1, C]
        y = (y_att + y_mean + y_cov) / 3.0
        return (y, attn, None) if return_attention else (y, None, None)


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, mode="ensemble")
