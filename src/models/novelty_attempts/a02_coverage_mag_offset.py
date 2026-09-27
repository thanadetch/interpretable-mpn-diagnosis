"""a02 — bag-magnitude noise-floor offset (ablation companion to a01).

Hint targeted: same as a01_coverage_frac_offset (BH1_g0_g1_absence).

Role: ablation companion. Identical architecture, identical parameter
count, identical offset machinery as a01 — the ONLY difference is how
the coverage scalar c ∈ [0, 1] is computed:

    a01 (main):     c = mean_i  σ((‖h_i‖₂ − τ) / β)     # fraction-of-active
    a02 (ablation): c = σ((mean_i ‖h_i‖₂ − τ) / β)      # bag-level magnitude

The active ingredient under test in a01 is the *per-patch* coverage
(fraction of fibre-positive patches, matches the clinical G3 criterion).
a02 replaces it with a single bag-level energy scalar passed through the
same sigmoid. If a01 beats both the baseline and a02, then the win is
specifically from "count of active patches", not from "subtract an
offset whenever overall bag energy is low".

Kill criterion: report regardless of outcome; this run exists to
contrast with a01. (No standalone QWK threshold.)

Param count: 197,250 (baseline) + 3 scalars = 197,253.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    import math

    return math.log(math.expm1(y))


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,
        tau_init: float = 1.0,
        beta_init: float = 1.0,
        lambda_init: float = 1e-3,
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

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        self.lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))

    def _coverage(self, h: torch.Tensor) -> torch.Tensor:
        """Bag-magnitude scalar c ∈ [0, 1] (ablation form)."""
        norms = torch.linalg.vector_norm(h, ord=2, dim=-1)  # [N]
        beta = F.softplus(self.beta_raw).clamp(min=1e-4)
        return torch.sigmoid((norms.mean() - self.tau) / beta)  # scalar

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
        s = self.classifier(bag)

        c = self._coverage(h)
        lam = F.softplus(self.lambda_raw)
        y = s - lam * (1.0 - c)

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
    tau_init=1.0,
    beta_init=1.0,
    lambda_init=1e-3,
)

