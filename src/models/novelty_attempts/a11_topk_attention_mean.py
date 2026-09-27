"""a11 — top-K attention-weighted mean (H7, K=5).

Hint targeted (§7 of NOVELTY_NOTES.md):
    H7_topk_attention_mean. Mechanism: REPLACE the full-bag softmax-mean
    of ABMIL with an attention-weighted mean computed over only
    the top-K scoring patches. K is a fixed hyperparameter (here K=5),
    so the effective bag size is constant across N. This is a STRUCTURAL
    REPLACEMENT of the aggregator, not an additive mechanism on top of
    it -- which is why it sits outside the dead-end family identified
    across DE11..DE15 ("every additive mechanism on top of ABMIL
    overfits val").

Why this is the right next step (post-batch-5 diagnosis):
    - Batches 1..5 all peaked val_qwk in 0.78-0.80 while baseline sits at
      0.8182. The common failure: extra params + same softmax-mean
      structure → val ceiling held by softmax dilution on large bags.
    - §5.5 confirms G0 bags are bag-size outliers (mean 53.4, max 112 vs
      ~40 for G1/G2/G3). Full-bag softmax-mean dilutes the signal across
      many irrelevant patches in big bags. Top-K restricts attention to
      the K patches the model thinks matter most, INDEPENDENT of N.
    - Same param count as `simple` baseline (197,250) -- removes
      "extra-capacity overfit" as a confounder. Any val_qwk delta vs
      baseline is purely from the structural pooling change.

Mechanism (exact):
    h_i        = bottleneck(features_i)                       # [N, 128]
    z_i        = W(V(h_i) ⊙ U(h_i)).squeeze(-1)               # baseline gated-attn logit
    if N > K:
        idx        = topk_indices(z, K)                       # K best-scoring patches
        z_top      = z[idx]                                   # [K]
        h_top      = h[idx]                                   # [K, 128]
        alpha_top  = softmax(z_top)                           # [K]  (renormalised)
        bag        = sum_k alpha_top_k * h_top_k              # [128]
    else:
        bag        = softmax(z) @ h                           # fall back to full bag
    y          = clamp(Linear(bag), 0, 3)

Pathology rationale:
    Pathologists grade reticulin fibrosis by inspecting the most fibrous
    regions in the ROI -- they do not average across the whole tissue.
    Top-K mimics this "find the worst K patches and grade based on them"
    operating point, which is also bag-size invariant.

Ablation companion: a12_topk_attention_mean_k1.py -- identical wiring
    but K=1 (pure argmax patch, no averaging). If a11 (K=5) beats a12
    (K=1), the active ingredient is "averaging across multiple worst
    patches", not just "pick the single worst patch". If both beat
    baseline, top-K replacement is the family answer. If both fail, the
    structural-replacement direction is dead and the next move is a
    second-order interaction family (e.g., H2 bottom-q noise floor).

Kill criterion: abandon if a11 val_qwk < 0.81 at seed=2.

Param count: same as ABMIL = 197,250 (no new params).

Determinism note: `torch.topk` is deterministic on equal scores within a
single device run; combined with seed=2 and frozen features this is
fully reproducible. Permutation-invariance: top-K selects on score
values, not on indices -- input permutation permutes the selected set
identically, so the final pooled vector is unchanged.
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
        K: int = 5,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert K >= 1, "K must be >= 1"
        self.clamp_output = clamp_output
        self.K = int(K)

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

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (trainer always passes a single bag).
        h = self.bottleneck(features)  # [N, hidden]
        n = h.shape[0]

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_logits = self.attention_W(V * U).squeeze(-1)  # [N]

        k = min(self.K, n)
        if k < n:
            # Select top-K by raw attention logits, renormalise via softmax
            # restricted to the top-K.
            top_logits, top_idx = torch.topk(attn_logits, k)
            top_h = h.index_select(0, top_idx)  # [K, hidden]
            top_alpha = F.softmax(top_logits, dim=0)  # [K]
            bag = torch.mm(top_alpha.unsqueeze(0), top_h).squeeze(0)  # [hidden]

            if return_attention:
                # Scatter the top-K attention back into an [N] vector for logging.
                full_attn = torch.zeros_like(attn_logits)
                full_attn.scatter_(0, top_idx, top_alpha)
                attention = full_attn
            else:
                attention = None
        else:
            # Bag is smaller than (or equal to) K -- fall back to full-bag softmax.
            attention = F.softmax(attn_logits, dim=0)
            bag = torch.mm(attention.unsqueeze(0), h).squeeze(0)

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
    K=5,
)

