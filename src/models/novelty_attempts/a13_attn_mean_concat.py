"""a13 — attention-pool ⊕ mean-pool concat (H8, regularization-via-ensemble).

Hint targeted (§7 of NOVELTY_NOTES.md):
    H8_attn_mean_concat_ensemble. Mechanism: compute BOTH the gated-
    attention-weighted pool AND the plain mean of the same bottleneck
    features, CONCATENATE them as a [256]-d bag representation, and
    feed to a single Linear(256, 1) head.

Why this is the right next step (post-batch-6 diagnosis):
    Batches 1..6 share one signature: every novelty's train_qwk climbs
    to 0.97-1.0 within 5-10 epochs while val_qwk peaks at 0.78-0.80,
    then degrades. This is overfit, not bad pooling. The baseline at
    0.8182 may be at a sweet spot of capacity. Adding ANY adaptive
    mechanism on top (DE11..15) or replacing the softmax-mean with a
    smaller-effective-bag variant (DE_topk family closing in batch 6)
    just gives more DOF to chase val-set noise.

    H8 attacks overfit DIRECTLY by pairing the adaptive attention pool
    with a NON-ADAPTIVE mean pool inside a single fused representation.
    The mean branch has zero per-patch DOF -- it CANNOT overfit per-
    patch attention noise -- so the head can lean on whichever branch
    generalises better. The Linear(256, 1) head adds only +128 params
    over baseline (197,378 vs 197,250).

Mechanism (exact):
    h_i        = bottleneck(features_i)                       # [N, 128]
    z_i        = W(V(h_i) ⊙ U(h_i)).squeeze(-1)               # baseline gated-attn logit
    alpha      = softmax(z, dim=0)                            # [N]
    bag_attn   = alpha @ h                                    # [128]
    bag_mean   = h.mean(dim=0)                                # [128]  -- no params
    bag        = concat(bag_attn, bag_mean)                   # [256]
    y          = clamp(Linear(256, 1)(bag), 0, 3)             # head

Pathology rationale:
    Attention captures "which patches matter most" (foreground / fibrous
    signal). Mean captures "what does the bag look like on average"
    (tissue-density baseline). Both are clinically relevant: pathologists
    grade by inspecting the worst regions AND comparing to overall
    tissue density. The two views are complementary and information-
    theoretically non-redundant for ordinal grading.

Ablation companion: a14_mean_only.py -- bag = mean(h), no attention
    branch at all (Linear(128, 1) head). This is essentially the legacy
    `mean_pool + bottleneck` baseline; we expect it to underperform.
    If a13 beats a14 AND beats baseline (0.8182), the attention+mean
    COMBINATION is the active ingredient. If a13 beats a14 but not
    baseline, the attention branch alone is doing the work (and the
    mean concat is dead weight).

Kill criterion: abandon if a13 val_qwk < 0.81 at seed=2.

Param count: baseline 197,250 + Linear head delta (256→1 vs 128→1)
    = 197,250 + 128 = 197,378.
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
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        self.clamp_output = clamp_output
        self.use_attention = use_attention

        # Bottleneck (identical to SimpleGatedMIL).
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        if use_attention:
            # Gated attention (identical to SimpleGatedMIL).
            self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            self.attention_W = nn.Linear(hidden_dim, 1)
            head_dim = 2 * hidden_dim  # concat(bag_attn, bag_mean)
        else:
            head_dim = hidden_dim  # mean only

        self.classifier = nn.Linear(head_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: [N, D] (trainer always passes a single bag).
        h = self.bottleneck(features)  # [N, hidden]

        bag_mean = h.mean(dim=0)  # [hidden]  -- no params, no per-patch DOF

        attention: Optional[torch.Tensor] = None
        if self.use_attention:
            V = self.attention_V(h)
            U = self.attention_U(h)
            attn_logits = self.attention_W(V * U).squeeze(-1)  # [N]
            attention = F.softmax(attn_logits, dim=0)  # [N]
            bag_attn = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden]
            bag = torch.cat([bag_attn, bag_mean], dim=-1)  # [2*hidden]
        else:
            bag = bag_mean

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
    use_attention=True,
)

