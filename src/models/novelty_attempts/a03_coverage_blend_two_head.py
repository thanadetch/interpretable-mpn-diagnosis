"""a03 — two-sided coverage-gated blend between range-restricted specialists.

Hint targeted (from §7 of NOVELTY_NOTES.md):
    H6_two_sided_coverage_blend — directly addresses the failure mode of
    DE11_coverage_offset_additive (batch 1). a01 lifted val G0 recall
    9.5%→28.6% via a coverage-driven *additive* downward offset, but the
    same offset simultaneously hurt G1 (90.6→86.8) and especially G2
    (66.3→60.0). The offset is monotone-downward only and so cannot
    distinguish "G0 needs to go DOWN" from "G2 needs to stay PUT" once
    coverage is intermediate.

Hypothesis: replace the additive offset with a *blend between two
    range-restricted regression heads*. A low-coverage specialist h_low
    is constrained to output in [0, 1.5] (G0/G1 territory); a
    high-coverage specialist h_high is constrained to output in [1.5, 3]
    (G2/G3 territory). The same coverage scalar c ∈ [0, 1] picks between
    them as a soft blend:
        ŷ = (1 − c) · h_low + c · h_high          ∈ [0, 3]
    For low-coverage bags (G0/G1) the answer is locked into the lower
    half of the ordinal; for high-coverage bags (G2/G3) it is locked
    into the upper half. There is no clamp() needed because both
    components are already bounded.

Mechanism:
    h_i        = bottleneck(features)                              # same as baseline
    α_i        = softmax(W (V tanh ⊙ U sigmoid)(h_i))              # same as baseline
    bag        = Σ_i α_i h_i                                       # same as baseline
    n_i        = ‖h_i‖₂
    c          = mean_i σ((n_i − τ) / β)                           # per-patch coverage, τ, β learnable
    z_low      = Linear_low(bag);   h_low  = 1.5 · σ(z_low)        # ∈ [0, 1.5]
    z_high     = Linear_high(bag);  h_high = 1.5 + 1.5 · σ(z_high) # ∈ [1.5, 3]
    ŷ          = (1 − c) · h_low + c · h_high                      # ∈ [0, 3]
where (τ, β) are learnable scalars; β = softplus(β_raw) keeps β > 0.

Ablation companion: a04_constant_blend_two_head.py — identical wiring
    except c is replaced by σ(γ), where γ is a single learnable scalar
    that does NOT depend on the bag features. Init γ = 0 so c = 0.5
    throughout training (the blend is a fixed 50/50 mix at start; γ can
    drift but it cannot encode any per-bag coverage signal). This
    isolates *coverage signal* as the active ingredient — if a03 beats
    a04, the per-bag coverage is doing useful work; if both fail, the
    two-head range split alone is not enough.

Kill criterion: abandon this hypothesis if a03 val_qwk < 0.81 at seed=2
    (same threshold used for a01 in batch 1 — i.e. fails to even match
    the locked baseline).

Param count: baseline 197,250 has a single Linear(128 → 1) classifier
    head (129 params). a03 replaces it with TWO Linear(128 → 1) heads
    (258 params) and adds 2 learnable scalars (τ, β). Net change:
    +258 − 129 + 2 = +131 params. Total: 197,381.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    """Return x such that softplus(x) = y, for y > 0."""
    import math

    return math.log(math.expm1(y))


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        clamp_output: bool = True,  # kept for API parity; outputs are already bounded
        tau_init: float = 1.0,
        beta_init: float = 1.0,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output

        # Same bottleneck as ABMIL.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Same gated attention as ABMIL.
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Two range-restricted regression heads (one for each ordinal half).
        self.head_low = nn.Linear(hidden_dim, num_classes)
        self.head_high = nn.Linear(hidden_dim, num_classes)

        # Coverage-gate parameters (2 learnable scalars; per-bag c is computed in fwd).
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))

    def _coverage(self, h: torch.Tensor) -> torch.Tensor:
        """Per-patch coverage fraction c ∈ [0, 1] from bottleneck h [N, D]."""
        norms = torch.linalg.vector_norm(h, ord=2, dim=-1)  # [N]
        beta = F.softplus(self.beta_raw).clamp(min=1e-4)
        p = torch.sigmoid((norms - self.tau) / beta)  # [N]
        return p.mean()  # scalar

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D]
        h = self.bottleneck(features)

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden]

        c = self._coverage(h)  # scalar in [0, 1]
        h_low = 1.5 * torch.sigmoid(self.head_low(bag))          # ∈ [0, 1.5]
        h_high = 1.5 + 1.5 * torch.sigmoid(self.head_high(bag))  # ∈ [1.5, 3]

        y = (1.0 - c) * h_low + c * h_high  # ∈ [0, 3]
        # No explicit clamp needed; kept for safety against fp drift.
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
)

