"""a68 — FUSED gated-attention MIL (decisive backbone-fusion signal test).

This is a faithful, capacity-matched clone of the locked baseline
``ABMIL`` (Linear(D->128)+ReLU+Dropout(0.5) bottleneck -> Ilse gated
attention -> attention-weighted mean -> Linear(128->1)), with ONE change: the
input dimension is taken from ``feature_dim_override`` instead of the trainer's
forced ``input_dim`` (src/train_grading_reti.py:1017 hard-sets
``kwargs['input_dim']`` from the backbone, which would otherwise build a
1280-wide bottleneck and crash on fused 2816-d bags).

Purpose: answer the make-or-break question for the fusion lever — *does
concatenating complementary frozen backbones (Virchow2 1280 + UNI2-h 1536 =
2816) add any grade signal to the STRONGEST existing aggregator?* Holding the
aggregator fixed and changing only the input space isolates fusion's
contribution. Companion ablation = a69 (same model, single-backbone 1280-d).

Run (fused):   --backbone virchow2 --data_root data_fused --novelty_id a68_fused_gated_mil
Run (ablation):--backbone virchow2 --data_root data       --novelty_id a69_single_gated_mil
Evaluate PAIRED across folds {0,1,2,3,42} (seed = patient fold).

No ||h|| weighting; permutation- and bag-size-invariant; deterministic at
inference. The metrics/border-white logit-bias of the baseline is dropped
because the trainer never passes ``metrics`` to novelty modules.
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
        feature_dim_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        # The trainer forces kwargs["input_dim"] from the backbone config, so we
        # use feature_dim_override (which it does NOT touch) when provided.
        eff_dim = feature_dim_override if feature_dim_override is not None else input_dim
        self.eff_dim = eff_dim

        self.bottleneck = nn.Sequential(
            nn.Linear(eff_dim, hidden_dim),
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
        h = self.bottleneck(features)  # [N, hidden]
        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]
        aggregated = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden]
        logits = self.classifier(aggregated)  # [num_classes]
        if return_attention:
            return logits, attention, None
        return logits, None, None


KWARGS = dict(
    input_dim=1280,            # ignored — trainer overwrites this; see feature_dim_override
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    feature_dim_override=2816,  # Virchow2 (1280) + UNI2-h (1536); run with --data_root data_fused
)
