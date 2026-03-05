"""
DualStreamMIL: Two-stream Gated-Attention MIL that fuses local morphology
(attention-weighted pooling) with global tissue composition (density-weighted
mean pooling) for improved PV recall.

Architecture:
    Bottleneck: Linear(input_dim, hidden_dim) -> ReLU -> Dropout
    Stream A (Local):  Gated Attention (Tanh/Sigmoid) -> attention-weighted pooling
    Stream B (Global): Conditional density-weighted or plain mean pooling
    Classifier: Linear(hidden_dim, num_classes)  [residual fusion]

Returns (logits, attention_weights, None) for compatibility with existing
training and evaluation loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DualStreamMIL(nn.Module):
    """
    Dual-Stream Gated-Attention MIL.

    Stream A captures salient local morphology via standard gated attention.
    Stream B captures global tissue composition via density-weighted mean pooling
    (when metrics are provided) or plain mean pooling (fallback).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        fixed_gamma: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Shared bottleneck projection
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Stream A: Gated attention
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Fusion gate: fixed constant or learnable parameter
        if fixed_gamma is not None:
            self.register_buffer("gamma", torch.tensor(fixed_gamma))
            self.gamma_param = None
        else:
            self.gamma_param = nn.Parameter(torch.tensor(-4.0))
            self.gamma = None  # computed in forward

        # Classifier: residual-fused representation
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            features: Instance features [N, D].
            return_attention: Unused, kept for API compatibility.
            metrics: Optional dict with patch-level quality metrics.
                     If present and contains ``"bad"``, Stream B uses
                     density-weighted mean pooling; otherwise plain mean.

        Returns:
            logits: Bag-level logits [C].
            attention: Attention weights [N].
            None: Placeholder for compatibility.
        """
        # Shared bottleneck
        h = self.bottleneck(features)  # [N, hidden_dim]

        # ── Stream A: Local (Gated Attention) ─────────────────────────
        V = self.attention_V(h)  # [N, hidden_dim]
        U = self.attention_U(h)  # [N, hidden_dim]
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]
        z_local = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden_dim]

        # ── Stream B: Global (Composition) ────────────────────────────
        if metrics is not None and "bad" in metrics:
            bad = metrics["bad"].to(h.device).float()
            p_good = torch.clamp(1.0 - bad, min=1e-6, max=1.0)
            weights = p_good / p_good.sum()  # [N], normalised
            z_global = torch.mm(weights.unsqueeze(0), h).squeeze(0)  # [hidden_dim]
        else:
            z_global = h.mean(dim=0)  # [hidden_dim]

        # ── Residual Fusion & Classification ──────────────────────
        if self.gamma_param is not None:
            gamma = torch.sigmoid(self.gamma_param)
        else:
            gamma = self.gamma  # fixed constant (buffer)
        z_fused = z_local + gamma * z_global  # [hidden_dim]
        logits = self.classifier(z_fused)  # [C]

        return logits, attention, None
