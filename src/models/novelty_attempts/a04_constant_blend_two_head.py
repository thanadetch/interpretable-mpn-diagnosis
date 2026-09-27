"""a04 — ablation companion to a03. Two-head blend with a feature-independent gate.

Hint targeted: H6_two_sided_coverage_blend. Role: ablation companion.

Architecture is identical to a03_coverage_blend_two_head.py with one
single difference:

    a03 (main):     c = mean_i σ((‖h_i‖ − τ) / β)         # per-bag coverage signal
    a04 (ablation): c = σ(γ)                              # bag-INDEPENDENT scalar

γ is a single learnable parameter initialised to 0 (so c = 0.5 at the
start of training). The two-head range split (h_low ∈ [0, 1.5], h_high
∈ [1.5, 3]) is preserved. The blend can still adapt — but only as a
constant, not per-bag. This isolates the *per-bag coverage signal* as
the active ingredient under test:

    - If a03 beats a04 AND both gates, coverage is doing the work.
    - If a03 ≈ a04, the two-head split alone is enough; the coverage
      signal contributes nothing.
    - If both fail similarly, the two-head split itself doesn't help —
      add it to dead-ends.

Kill criterion: report regardless of outcome; this run only exists to
    contrast with a03. (No standalone QWK threshold.)

Param count: baseline 197,250 has a single Linear(128→1) head (129).
    a04 replaces it with two Linear(128→1) heads (258) and adds 1
    learnable scalar (γ). Net: +258 − 129 + 1 = +130 params.
    Total: 197,380 (≈ a03 = 197,381, differs by 1 param: τ/β vs γ).
"""
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
        clamp_output: bool = True,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        self.head_low = nn.Linear(hidden_dim, num_classes)
        self.head_high = nn.Linear(hidden_dim, num_classes)

        # Bag-independent blend gate (1 learnable scalar). c = sigma(gamma).
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_scores = self.attention_W(V * U).squeeze(-1)
        attention = F.softmax(attn_scores, dim=0)

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)

        c = torch.sigmoid(self.gamma)  # scalar, no per-bag signal
        h_low = 1.5 * torch.sigmoid(self.head_low(bag))
        h_high = 1.5 + 1.5 * torch.sigmoid(self.head_high(bag))

        y = (1.0 - c) * h_low + c * h_high

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attention, None
        return y, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    gamma_init=0.0,
)

