"""a15 — bag-subsampling augmentation (H9, training-time only).

Hint targeted (§7 of NOVELTY_NOTES.md):
    H9_bag_subsample_aug. Mechanism: at TRAINING time, if a bag has
    more than K_max patches, randomly subsample K_max patches (without
    replacement) before passing to the ABMIL aggregator. At
    EVAL time, always use the full bag. No architectural change; no
    extra params. The aggregator itself is bit-for-bit identical to the
    locked baseline (SimpleGatedMIL).

Why this is the right next step (post-batch-7 diagnosis):
    Six consecutive batches (1..6 architecture; 7 ensemble) all showed
    the same signature: train_qwk → 0.97-1.0 within 5-10 epochs while
    val_qwk peaks at 0.78-0.80 then degrades. This is OVERFIT, not
    architectural mismatch. With only 857 train bags, any added or
    rearchitected aggregator capacity gets memorised.

    H9 attacks overfit ORTHOGONALLY: it leaves the architecture alone
    and instead INCREASES the effective training distribution by
    sampling random sub-bags every epoch. A G0 bag with 100 patches
    becomes a different 24-patch view every step — the model can no
    longer memorise "this exact 100-patch bag is G0" because it never
    sees the same 100-patch bag twice. Critically, this directly
    attacks the §5.5 bag-size shortcut: by capping training bags at
    K_max=24 (well below the G0 mean of 53.4 and max 112) while leaving
    all other grades' bags MOSTLY unchanged (G1/G2/G3 medians are 40-42,
    so they get subsampled too but less aggressively), the "many
    patches ⇒ G0" spurious correlation is broken.

    At inference the full bag is used, so the aggregator sees its
    natural input distribution — only the training distribution is
    altered. This is the classical data-augmentation trick, applied at
    bag granularity.

Mechanism (exact):
    if self.training and N > K_max:
        idx       = randperm(N)[:K_max]                  # uniform random K_max patches
        features  = features.index_select(0, idx)        # [K_max, D]
    h            = bottleneck(features)                  # ABMIL baseline below
    V            = tanh(W_V(h));  U = sigmoid(W_U(h))
    z            = W(V * U).squeeze(-1)
    alpha        = softmax(z, dim=0)
    bag          = alpha @ h
    y            = clamp(Linear(bag), 0, 3)

Pathology rationale:
    A pathologist grading reticulin fibrosis does NOT need to see every
    patch of the ROI to assign a grade — they spot-check representative
    regions. Forcing the model to grade from 24 randomly-chosen patches
    teaches it to be ROBUST to bag composition, which is precisely
    the right inductive bias for a clinical grading system.

Ablation companion: a16_bag_subsample_aug_off.py — identical wiring
    but K_max set to a value > any bag in the dataset (effectively
    disables subsampling). This is bit-for-bit the ABMIL
    baseline. If a15 beats a16, bag subsampling is the active
    ingredient (and a16 should reproduce baseline val_qwk = 0.8182 to
    within noise, serving as a positive control).

Kill criterion: abandon if a15 val_qwk < 0.81 at seed=2.

Param count: same as ABMIL = 197,250 (no new params).

Determinism note: subsampling uses `torch.randperm(N, device=features.device)`
which is seeded by the trainer's global seed=2. Each (epoch, bag) pair
sees a deterministic sub-bag. At eval, `self.training` is False so the
full bag is always used — no test-time stochasticity.
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
        K_max: int = 24,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "regression head only (num_classes=1)"
        assert K_max >= 1, "K_max must be >= 1"
        self.clamp_output = clamp_output
        self.K_max = int(K_max)

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
        # features: [N, D].
        if self.training and features.shape[0] > self.K_max:
            idx = torch.randperm(features.shape[0], device=features.device)[: self.K_max]
            features = features.index_select(0, idx)

        h = self.bottleneck(features)  # [N', hidden]

        V = self.attention_V(h)
        U = self.attention_U(h)
        attn_logits = self.attention_W(V * U).squeeze(-1)  # [N']
        attention = F.softmax(attn_logits, dim=0)  # [N']

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
    K_max=24,
)

