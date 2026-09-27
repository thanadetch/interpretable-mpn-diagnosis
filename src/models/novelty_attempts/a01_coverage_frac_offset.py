"""a01 — coverage-fraction noise-floor offset on the ABMIL baseline.

Hint targeted (val CM of locked baseline
    experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342,
    inspected directly because leader_diagnostics.py cannot join the
    legacy-baseline val_predictions onto the full-data split):
    Hint BH1_g0_g1_absence — G0↔G1 confusions are 40.4% of val errors
    (G0 recall 9.5% on val; 17/21 G0 ROIs misrouted to G1).

Hypothesis (H1_coverage_via_temperature, refined):
    ABMIL's attention-weighted bag mean is *coverage-blind* — a few
    fibre-like patches in a G0 ROI dominate the bag score because softmax
    attention is sharp. Adding a soft "fraction-of-active-patches" prior
    should pull bags with sparse activity toward grade 0 without touching
    bags whose patches are uniformly active (G3).

    Mechanism:
        h_i        = bottleneck(features)   # same as baseline
        α_i        = softmax(W(V tanh ⊙ U sigmoid)(h_i))   # same as baseline
        s          = Linear(128→1)(Σ_i α_i h_i)             # baseline score
        n_i        = ‖h_i‖₂                                 # per-patch energy
        c          = mean_i σ((n_i − τ) / β)               # *fraction* active
        ŷ          = clamp( s − λ · (1 − c),  0, 3 )
    where (τ, β, λ) are learnable scalars; β = softplus(β_raw) keeps β > 0;
    λ = softplus(λ_raw) keeps the offset non-negative (it can only pull
    predictions down, never up). All three are initialised so the network
    matches the baseline output at step 0 (λ ≈ 0 ⇒ ŷ ≈ s).

Ablation companion: a02_coverage_mag_offset.py — identical wiring except
    c = σ((mean_i n_i − τ) / β) is a *bag-level magnitude* scalar instead of
    the per-patch coverage fraction. Same parameter count, same offset
    machinery, same initialisation. If a01 beats both the baseline and a02,
    the active ingredient is specifically the "fraction of active patches"
    prior (BH1), not just an offset based on overall bag energy.

Kill criterion: abandon this family if a01 val_qwk < 0.81 at seed=2
    (i.e. fails to even match the locked baseline).

Param count: 197,250 (baseline SimpleGatedMIL) + 3 scalars = 197,253.
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
        clamp_output: bool = True,
        tau_init: float = 1.0,
        beta_init: float = 1.0,
        lambda_init: float = 1e-3,
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

        # Same regression head as ABMIL.
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Coverage-fraction offset parameters (3 learnable scalars).
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        self.lambda_raw = nn.Parameter(torch.tensor(_inv_softplus(lambda_init)))

    # --------------------------------------------------------------- helpers
    def _coverage(self, h: torch.Tensor) -> torch.Tensor:
        """Per-patch coverage fraction c ∈ [0, 1] from bottleneck h [N, D]."""
        norms = torch.linalg.vector_norm(h, ord=2, dim=-1)  # [N]
        beta = F.softplus(self.beta_raw).clamp(min=1e-4)
        p = torch.sigmoid((norms - self.tau) / beta)  # [N]
        return p.mean()  # scalar

    # ------------------------------------------------------------------ fwd
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (trainer always passes single bag)
        h = self.bottleneck(features)  # [N, hidden]

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_scores = self.attention_W(V * U).squeeze(-1)  # [N]
        attention = F.softmax(attn_scores, dim=0)  # [N]

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden]
        s = self.classifier(bag)  # [num_classes] == [1]

        c = self._coverage(h)  # scalar in [0, 1]
        lam = F.softplus(self.lambda_raw)  # ≥ 0
        y = s - lam * (1.0 - c)  # [1]

        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if return_attention:
            return y, attention, None
        return y, None, None


# Trainer overrides input_dim / num_classes to match the chosen backbone.
KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau_init=1.0,
    beta_init=1.0,
    lambda_init=1e-3,
)

