"""a122 — pure bone-avoidance sweep: force the attention away from bone by a FIXED
strength lambda, with NO fibrosis boost, to expose the bone<->grade tension.

    attn_logits_i = gated_score_i - lambda * bone_z_i

lambda is read from env var A122_LAMBDA (default 0 = exactly the baseline gated
attention). Sweeping lambda traces a tradeoff curve: as lambda grows the attention
de-correlates from bone (more faithful to the bone-avoidance prior) but — because
the grade-relevant fibrosis signal is entangled with bone (per-patch concept
corr ~0.79) — the grade accuracy is expected to DROP. That curve is the thesis's
interpretability centerpiece: you cannot avoid bone without sacrificing grade
signal on frozen FM features.

Input: 1282-d bag = [virchow2 1280 | bone | fibrosis] (data_bonefib). forward
reads ONLY 'features'. RAW logit. Deterministic at inference.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_BONE_MEAN, _BONE_STD = -0.0486, 0.0193


def _lambda() -> float:
    try:
        return float(os.environ.get("A122_LAMBDA", "0.0"))
    except ValueError:
        return 0.0


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.feat_dim = input_dim
        self.lam = _lambda()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        D = self.feat_dim
        feats = features[:, :D]
        h = self.bottleneck(feats)
        gated = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        if self.lam != 0.0 and features.shape[1] >= D + 1:
            bone = (features[:, D] - _BONE_MEAN) / _BONE_STD
            gated = gated - self.lam * bone
        attn = F.softmax(gated, dim=0)
        agg = torch.mm(attn.unsqueeze(0), h).squeeze(0)
        out = self.classifier(agg)
        if return_attention:
            return out, attn, None
        return out, None, None


KWARGS = dict(input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5)
