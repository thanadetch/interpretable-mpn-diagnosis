"""a47 — ordinal cumulative-link head on ABMIL aggregator (H23, batch 23).

Philosophy bucket: ordinal_head

Hint targeted (NOVELTY_SEARCH_PLAYBOOK.md §12.4): first batch in the
`ordinal_head` bucket. The 42 prior aNN runs all used a plain
`Linear(128, 1) + clamp[0, 3]` regression head that treats the four
fibrosis grades G0/G1/G2/G3 as positions on a real-valued axis. This
ignores the *ordinal* nature of the target: a G1↔G2 confusion is
penalised identically to a G1↔G3 confusion only if the predicted score
happens to land at the same distance, and the regression head has no
built-in inductive bias that grade boundaries should exist.

Hypothesis (H23_ordinal_cumulative_link_head)
---------------------------------------------
Replace the regression head with a **cumulative-link parameterisation
of the ordinal class probabilities**, while keeping the rest of the
ABMIL architecture identical to the locked baseline. The bag
representation produces a single latent score ``s = w^T h``; three
learnable, **monotonically ordered** thresholds
``τ_0 < τ_1 < τ_2`` partition the score axis into 4 ordinal regions
G0/G1/G2/G3.

The bag-level prediction is the **expected ordinal class**:

    ŷ = Σ_{k=0}^{2} σ(s − τ_k)                ∈ [0, 3]

i.e. the sum of cumulative tail probabilities ``P(y > k | s)``. This
expression is well-defined as a scalar in ``[0, 3]`` and is therefore
compatible with the trainer's locked ``--formulation regression`` +
``SmoothL1Loss`` pipeline — no trainer change required (playbook §0
hard rule, §10 anti-pattern).

Why this should help on top of vanilla regression
--------------------------------------------------
1. **Implicit ordinal calibration.** SmoothL1 on ŷ rewards the model
   for placing ``s`` such that *all three* thresholds are crossed in
   the correct order for high-grade bags and *none* are crossed for
   G0 bags. The regression baseline, in contrast, can satisfy SmoothL1
   by any monotone mapping ``s → grade``; there is no benefit to
   making ``s`` near-piecewise-constant on each grade.
2. **Pathology motivation.** Reticulin fibrosis grading is *defined*
   ordinally (G0 = no fibrosis, G3 = diffuse + coarse fibers); the
   thresholds ``τ_k`` correspond directly to the pathologist's
   diagnostic boundaries between consecutive grades. Making them
   explicit and learnable yields an interpretable model whose
   thresholds can be inspected post-hoc.
3. **Tie-zone robustness.** The 42 prior aNN modules cluster in
   ``val_qwk ∈ [0.78, 0.81]`` (post-mortem §12). A different inductive
   bias (head, not aggregator) is the obvious axis to perturb next —
   this is the first ``*_head`` family to be evaluated.

Mechanism (exact)
-----------------
1. Bottleneck    h_i = Dropout(ReLU(W_b h_i^raw))        ∈ R^128
2. Gated attn    a_i = w^T (tanh(V h_i) ⊙ σ(U h_i))      ∈ R
                 α   = softmax(a)                         ∈ Δ^N
3. Bag rep       b   = Σ_i α_i h_i                        ∈ R^128
4. Score         s   = w_s^T b + b_s                      ∈ R
5. Thresholds    τ_0 = τ_0_raw
                 τ_1 = τ_0 + softplus(δ_1)
                 τ_2 = τ_1 + softplus(δ_2)
   (so τ_0 < τ_1 < τ_2 is guaranteed at every step of training)
6. Prediction    ŷ  = Σ_{k=0}^{2} σ(s − τ_k)             ∈ [0, 3]
   (no extra clamp needed; the sum is in [0, 3] by construction)

Initialisation. ``τ_0 = 0.5, δ_1 = δ_2 = log(e − 1) ≈ 0.5413`` so that
``softplus(δ_k) = 1`` and ``τ = (0.5, 1.5, 2.5)`` at init — the
midpoints between consecutive grade integers, matching how a vanilla
regression baseline naturally rounds at half-integers.

Ablation companion
------------------
``a48_gated_attention_regression_head.py`` (Philosophy bucket:
ordinal_head). Identical aggregator, replaces the cumulative-link head
with ``ŷ = clamp(Linear(128, 1)(b), 0, 3)`` — the standard regression
head. Functions as both a positive control (should reproduce the
locked baseline closely, since the aggregator is bit-for-bit
ABMIL minus the `attention_logit_bias` border-white prior)
and as the **active-ingredient isolator** for the cumulative-link head.

Predictions
-----------
- a47 > a48 ≥ baseline → ordinal head is the active ingredient.
- a47 ≈ a48 ≈ baseline → cumulative-link head buys nothing on this
  dataset; ``ordinal_head`` bucket family essentially dead.
- a47 < a48 → ordinal head hurts (over-rigid). Worth reporting as a
  clean negative result for the systematic study (Ch. 4.3).

Kill criterion
--------------
Family-level: if BOTH a47 and a48 have val_qwk < 0.79 at seed=2,
declare the H23 family dead; move to `multi_scale_fusion` or
`backbone_fusion` per §12.4. Main-only: if a47 ≤ a48 by ≥ 0.005
val_qwk, the ordinal head is not pulling its weight even with the
ablation tied; do not push compounds.

Param count (input_dim=1280, hidden_dim=128)
--------------------------------------------
    bottleneck   Linear(1280, 128) + bias  = 163,968
    attn_V       Linear(128, 128) + bias   =  16,512
    attn_U       Linear(128, 128) + bias   =  16,512
    attn_W       Linear(128, 1) + bias     =     129
    score        Linear(128, 1) + bias     =     129
    thresholds   (τ_0_raw, δ_1, δ_2)       =       3
    -------------------------------------------------
    total trainable                          = 197,253
    (= baseline 197,250 + 3 ordinal threshold scalars)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        # Threshold init so τ = (0.5, 1.5, 2.5) at start of training.
        tau0_init: float = 0.5,
        delta_init: float = 0.5413248,  # softplus(0.5413) ≈ 1.0
        use_ordinal_head: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "a47 is scalar-regression only (num_classes=1)."

        # ── Aggregator (same shape as SimpleGatedMIL, minus metrics-bias) ──
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)

        # ── Scalar score head (shared by ordinal + ablation control) ──
        self.score_head = nn.Linear(hidden_dim, 1)

        # ── Ordinal cumulative-link parameters ──
        self.use_ordinal_head = use_ordinal_head
        if use_ordinal_head:
            self.tau0_raw = nn.Parameter(torch.tensor(float(tau0_init)))
            self.delta_1_raw = nn.Parameter(torch.tensor(float(delta_init)))
            self.delta_2_raw = nn.Parameter(torch.tensor(float(delta_init)))

    def _aggregate(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """features: [B, N, D] → (bag_rep [B, H], attention [B, N])."""
        B, N, _ = features.shape
        h = self.bottleneck(features.reshape(B * N, -1)).reshape(B, N, -1)  # [B, N, H]
        V = self.attention_V(h)
        U = self.attention_U(h)
        a = self.attention_W(V * U).squeeze(-1)  # [B, N]
        alpha = F.softmax(a, dim=1)              # [B, N]
        bag = (alpha.unsqueeze(-1) * h).sum(dim=1)  # [B, H]
        return bag, alpha

    def _ordinal_predict(self, score: torch.Tensor) -> torch.Tensor:
        """score: [B, 1] → ŷ ∈ [0, 3] via cumulative-link expected class."""
        tau0 = self.tau0_raw
        tau1 = tau0 + F.softplus(self.delta_1_raw)
        tau2 = tau1 + F.softplus(self.delta_2_raw)
        # Broadcast: P(y > k) for k = 0, 1, 2
        p0 = torch.sigmoid(score - tau0)
        p1 = torch.sigmoid(score - tau1)
        p2 = torch.sigmoid(score - tau2)
        return p0 + p1 + p2  # [B, 1] in [0, 3]

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)
        bag, alpha = self._aggregate(features)
        score = self.score_head(bag)  # [B, 1]
        if self.use_ordinal_head:
            y = self._ordinal_predict(score)
        else:
            y = score.clamp(0.0, 3.0)
        if squeeze:
            y = y.squeeze(0)
            alpha = alpha.squeeze(0)
        return y, alpha, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    tau0_init=0.5,
    delta_init=0.5413248,
    use_ordinal_head=True,
)

