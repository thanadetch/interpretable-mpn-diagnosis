"""a05 — presence × severity multiplicative gate on ABMIL.

Hint targeted (§7 of NOVELTY_NOTES.md):
    H5_two_branch_presence_severity. Motivation reinforced by:
      - DE11 (batch 1): a01 additive coverage offset lifted val G0 recall
        9.5%→28.6% but dragged G2 down because the offset is monotone-
        downward only. The same coverage information should be encoded
        as a *multiplicative gate* that can drive ŷ→0 for G0 bags
        without affecting the scale for non-G0 bags.
      - H6 ablation (batch 2 a03/a04): the two-head sigmoid-bounded
        design suffered a gradient bottleneck. *Unbounded* heads with
        a final clamp avoid this. a05 uses one bounded scalar (presence)
        and one *unbounded* linear (severity).

Hypothesis: the ordinal grade can be factored into
        ŷ = presence · (severity_offset + severity_logit)
    where:
        presence  ∈ (0, 1) — soft G0-vs-≥G1 gate (one scalar bottleneck)
        severity_logit ∈ ℝ — unbounded "how far past G0" magnitude
    so that low-presence bags collapse toward 0 *multiplicatively*
    (fixing DE11) while keeping a free-flowing gradient on the severity
    side (fixing the H6/a03 saturation problem).

Mechanism:
    h_i        = bottleneck(features)                              # same as baseline
    α_i        = softmax(W (V tanh ⊙ U sigmoid)(h_i))              # same as baseline
    bag        = Σ_i α_i h_i                                       # same as baseline
    p_logit    = Linear_p(bag)                                     # scalar
    s_logit    = Linear_s(bag)                                     # scalar
    presence   = σ(p_logit)                                        # ∈ (0, 1)
    severity   = s_logit + severity_offset                         # ∈ ℝ, init around 1.5
    ŷ          = clamp(presence · severity, 0, 3)                  # ∈ [0, 3]
At init (linear heads near zero, severity_offset = 1.5),
    presence ≈ 0.5, severity ≈ 1.5 ⇒ ŷ ≈ 0.75 (near G1, matches dataset mean).
Both heads have well-conditioned gradients: presence's sigmoid sees
only one scalar of saturation pressure, and severity is fully linear
(no sigmoid bottleneck).

Ablation companion: a06_presence_constant.py — identical wiring except
    presence is `σ(γ)` with γ a feature-independent learnable scalar
    (init γ = 0 ⇒ presence = 0.5). Removes the *per-bag* presence
    signal but keeps the multiplicative structure. If a05 beats a06,
    the per-bag G0-vs-≥G1 decision is the active ingredient; if both
    win, structure alone is enough; if both fail, the family is dead.

Kill criterion: abandon if a05 val_qwk < 0.81 at seed=2 (i.e. fails to
    even match the locked baseline 0.8182).

Param count: baseline 197,250 has a single Linear(128→1) head (129
    params). a05 replaces it with TWO Linear(128→1) heads (258 params).
    No other learnable scalars. Net: +258 − 129 = +129 params.
    Total: 197,379.
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
        severity_offset: float = 1.5,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output
        self.severity_offset = float(severity_offset)

        # Bottleneck identical to ABMIL.
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated attention identical to ABMIL.
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        # Two regression heads sharing the bag rep.
        self.head_presence = nn.Linear(hidden_dim, num_classes)
        self.head_severity = nn.Linear(hidden_dim, num_classes)

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

        p_logit = self.head_presence(bag)                        # [1]
        s_logit = self.head_severity(bag)                        # [1]
        presence = torch.sigmoid(p_logit)                        # ∈ (0, 1)
        severity = s_logit + self.severity_offset                # ∈ ℝ, init ≈ 1.5
        y = presence * severity                                  # ∈ ℝ

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
    severity_offset=1.5,
)

