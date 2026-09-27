"""a09 — coverage-aware attention re-weight (H1, soft threshold).

Hint targeted (§7 of NOVELTY_NOTES.md):
    H1_coverage_via_temperature. Mechanism: per-patch coverage prior
        c_i = sigmoid((||h_i|| - tau) / beta)
    is injected as a *log-bias on the attention logit*, not as a gate on
    the prediction. The soft-mean becomes
        attn_i = softmax(z_i + alpha * log(c_i + eps))
    so coverage reweights which patches contribute to the bag rep, but
    the final regression head is the same unbounded Linear(128 -> 1) as
    the ABMIL baseline. No sigmoid ever sits between the bag
    representation and y_hat.

Why this is allowed after DE11/DE12/DE13:
    Those three dead-ends all routed the prediction through a bag-level
    sigmoid (additive coverage offset, range-restricted heads, multiplicative
    presence gate) and SmoothL1 could not push back through saturation.
    Here the sigmoid is purely per-patch and lives INSIDE a softmax
    normalisation -- its absolute scale is unidentified (softmax cancels
    constants), and gradient flow to the regression head is identical to
    the baseline. The only thing that changes is which patches the
    attention pools over.

Pathology motivation:
    Clinical G3 = "diffuse intersecting fibres covering a large fraction
    of the ROI". This is naturally a *coverage* quantity. The baseline's
    Ilse-style attention is free to fixate on 1-2 high-magnitude patches
    even when most of the bag is empty. Conditioning attention on a soft
    fibre-coverage prior pulls the bag rep towards "what fraction of the
    bag looks fibrous", which is the right pathology prior for G0 vs G1
    (absence) AND G2 vs G3 (density).

Mechanism (exact):
    h_i        = bottleneck(features)                       # [N, 128]
    z_i        = W(V(h_i) (.) U(h_i)).squeeze(-1)            # baseline gated-attn logit
    n_i        = ||h_i||_2                                   # per-patch magnitude
    c_i        = sigmoid((n_i - tau) / beta)                 # in (0, 1)
    alpha      = softplus(alpha_raw)                         # >= 0
    attn_i     = softmax_i(z_i + alpha * log(c_i + eps))     # coverage-reweighted
    bag        = sum_i attn_i * h_i
    y          = clamp(Linear(bag), 0, 3)

Initialisation:
    - tau = 1.0, beta = 1.0 (same as a01 / a02; the bottleneck-norm
      distribution is roughly O(1) under ReLU + Dropout at init).
    - alpha_raw chosen so softplus(alpha_raw) ~= 0.0 at init: with
      alpha = 0, the model is *exactly* the ABMIL baseline.
      We use alpha_init = 1e-3 -> alpha_raw ~= log(expm1(1e-3)) ~ -6.9.
      Training is free to scale alpha up.

Ablation companion: a10_coverage_attn_weight_hard.py — same module but
    beta is FIXED at 0.05 (~hard step). If a09 beats a10, the SOFT
    transition zone (gradients to tau via the full sigmoid slope) is the
    active ingredient. If both fail, the per-patch coverage-prior family
    is dead under simple + virchow2.

Kill criterion: abandon if a09 val_qwk < 0.81 at seed=2.

Param count:
    baseline 197,250 + 3 scalars (tau, beta_raw, alpha_raw) = 197,253.
"""
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
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
        alpha_init: float = 1e-3,
        beta_learnable: bool = True,
        beta_fixed: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output
        self.eps = eps
        self.beta_learnable = beta_learnable
        self.beta_fixed = float(beta_fixed)

        # Bottleneck + gated attention (identical to SimpleGatedMIL).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Coverage prior parameters.
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))
        if beta_learnable:
            self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
        else:
            # Fixed near-step threshold (used by ablation a10).
            self.register_buffer("_beta_const", torch.tensor(float(beta_fixed)))
        self.alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_init)))

    def _beta(self) -> torch.Tensor:
        if self.beta_learnable:
            return F.softplus(self.beta_raw).clamp(min=1e-4)
        return self._beta_const.clamp(min=1e-4)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (trainer always passes a single bag).
        h = self.bottleneck(features)  # [N, hidden]

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_logits = self.attention_W(V * U).squeeze(-1)  # [N]

        # Per-patch coverage prior in (0, 1).
        norms = torch.linalg.vector_norm(h, ord=2, dim=-1)  # [N]
        beta = self._beta()
        c = torch.sigmoid((norms - self.tau) / beta)  # [N]

        alpha = F.softplus(self.alpha_raw)  # >= 0 scalar
        log_c = torch.log(c + self.eps)  # in (-inf, 0]

        attention = F.softmax(attn_logits + alpha * log_c, dim=0)  # [N]

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden]
        y = self.classifier(bag)

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
    alpha_init=1e-3,
    beta_learnable=True,
)

