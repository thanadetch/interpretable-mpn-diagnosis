"""a17 — coverage-aware attention reweight + length-normalised softmax (H10, H1×H4 stack).

Hypothesis (H10_coverage_x_length_norm_stack): combine the two
mechanisms that individually beat the reproducible baseline (a16: val
0.7811 / test 0.9607) on val:
    - H1 (a09): per-patch coverage prior c_i = σ((||h_i||-τ)/β)
      injected as α·log(c_i) bias INSIDE the attention softmax.
      a09 gave val 0.7920 (+0.011 vs a16) / test 0.9517 (-0.009 vs a16).
    - H4 (a07): length-normalised softmax temperature T = c0/√N
      (c0 learnable, init so T ≈ 1 at median bag size N=40).
      a07 gave val 0.7868 (+0.006 vs a16) / test 0.9596 (-0.001 vs a16).

Both mechanisms reweight ATTENTION (no σ between bag rep and ŷ → no
DE11-13 gradient bottleneck), keep the unbounded Linear(128,1) head,
and add only a few scalar params. The combination attacks two distinct
failure modes:
    - coverage prior re-weights WHICH patches contribute,
    - length-normalised temperature scales HOW PEAKY the softmax is per
      bag size.
These are mechanistically orthogonal, so they may stack rather than
saturate. Worth trying because both individually beat a16 val and a07
is within 0.001 of a16 test (1 ROI of difference on 259-row test set).

Mechanism (exact):
    h_i        = bottleneck(features)                       # [N, 128]
    z_i        = W(V(h_i) ⊙ U(h_i)).squeeze(-1)             # baseline gated-attn logit
    n_i        = ||h_i||_2                                   # per-patch magnitude
    c_i        = sigmoid((n_i - τ) / β)                      # coverage prior (a09)
    T          = softplus(c0_raw) / sqrt(N)                  # length-norm temperature (a07)
    attn       = softmax((z + α·log(c+ε)) * T, dim=0)        # combined reweight
    bag        = sum_i attn_i * h_i
    y          = clamp(Linear(bag), 0, 3)

Initialisation: τ=1.0, β=1.0, α=1e-3 (≈0 → starts as pure a07), c0
such that T ≈ 1 at N=40 (c0_init = √40 ≈ 6.32). At init the model
behaves like a07 (≡ scaled SimpleGatedMIL); training is free to ramp
α to mix in the coverage prior.

Ablation companion: a18_length_norm_only.py — same wiring but α=0
(no coverage bias). This is bit-for-bit a07. Positive control: a18
should reproduce a07 val ≈ 0.7868 / test ≈ 0.9596.

Kill criterion: abandon if a17 val_qwk < a16's 0.7811 at seed=2.

Param count: baseline 197,250 + (τ, β_raw, α_raw, c0_raw) = 197,254.
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
        c0_init: float = 6.324555,
        use_coverage: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output
        self.use_coverage = use_coverage
        self.eps = eps

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Length-norm temperature (a07).
        self.c0_raw = nn.Parameter(torch.tensor(_inv_softplus(c0_init)))

        # Coverage prior (a09).
        if use_coverage:
            self.tau = nn.Parameter(torch.tensor(float(tau_init)))
            self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init)))
            self.alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha_init)))

    def _temperature(self, n: int) -> torch.Tensor:
        c0 = F.softplus(self.c0_raw)
        return c0 / math.sqrt(max(n, 1))

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)  # [N, hidden]
        n = h.shape[0]

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_logits = self.attention_W(V * U).squeeze(-1)  # [N]

        if self.use_coverage:
            norms = torch.linalg.vector_norm(h, ord=2, dim=-1)  # [N]
            beta = F.softplus(self.beta_raw).clamp(min=1e-4)
            c = torch.sigmoid((norms - self.tau) / beta)  # [N] in (0,1)
            alpha = F.softplus(self.alpha_raw)
            attn_logits = attn_logits + alpha * torch.log(c + self.eps)

        T = self._temperature(n)
        attention = F.softmax(attn_logits * T, dim=0)  # [N]

        bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)
        y = self.classifier(bag)

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
    alpha_init=1e-3,
    c0_init=6.324555,
    use_coverage=True,
)

