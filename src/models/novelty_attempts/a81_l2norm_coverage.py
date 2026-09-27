"""a81 — L2-normalized gated-attention MIL (strip the grade-irrelevant norm axis).

ANGLE (fresh, principled — not a re-run of any of a01-a80). The diagnostic
on all Virchow2 bags shows the per-patch feature *magnitude* ||f_i|| is
grade-uninformative (Spearman ~0 with grade; it tracks tissue-vs-background /
bone, a nuisance), while the *direction* of f_i carries the fibrosis signal
(coverage along the warm fibrosis axis separates the grade boundaries). The
baseline gated-attention aggregator consumes raw f_i, so its Linear(1280->128)
bottleneck must spend capacity disentangling the (large, grade-irrelevant)
magnitude from the (grade-relevant) direction on a tiny 214-ROI val cohort —
a nuisance degree of freedom that adds variance.

a81's single active ingredient: **L2-normalize every patch BEFORE aggregation**

        f_i  ->  f_i / ||f_i||                # unit-direction, magnitude removed

then run the EXACT baseline gated-attention bottleneck on the normalized bag:

        h_i   = Dropout(ReLU(Linear(1280->128) f_hat_i))     # baseline encoder
        a_i   = softmax_i( W ( tanh(V h_i) * sigmoid(U h_i) ) )  # Ilse gated attn
        z     = sum_i a_i h_i                                  # weighted mean
        y     = Linear(128->1) z                              # RAW logit (no clamp)

This is the cosine/projective view of the bag: the model sees only feature
DIRECTIONS, so the bottleneck no longer has to learn to be magnitude-invariant
from data. Hypothesis: removing the grade-irrelevant norm DOF lowers val<->test
variance (the stated bottleneck this round) without changing capacity.

WHY THIS IS NOT A TRIED MECHANISM:
  - It is NOT norm/rank salience (a45 and kin WEIGHTED patches BY ||h|| as an
    attention signal). a81 does the OPPOSITE — it DELETES ||f|| entirely and
    weights by the *learned* gated attention on directions. Norm is discarded,
    never used as a score.
  - It is NOT coverage / moments / quantiles / top-k / pairwise / dist-match /
    per-patch severity / consensus / PMA. The aggregator is the plain baseline
    gated-attention weighted MEAN — the ONLY change vs the baseline is the
    upstream L2-normalization of the inputs.
  - It does NOT weight by feature-norm anywhere (the explicit prohibition):
    L2-normalization removes the norm; it never multiplies/scores by it.

The warm fibrosis axis (data/prototypes_virchow2_reti_train_seed2.pt['axis'],
seed=2 train-only) is available but is intentionally NOT used here — a81's
whole claim is that direction-only baseline attention suffices once the norm
nuisance is gone; pulling in the axis would confound the isolation. (Loaded
read-only only if a future variant flips warm_init; default OFF, no leakage.)

ABLATION COMPANION: a82_l2norm_baseline_ablation.py imports THIS Model and
flips l2_normalize=False -> the plain baseline gated-attention MIL (operates on
raw f_i). a81 vs a82 isolates EXACTLY the active ingredient: "does explicitly
removing the grade-irrelevant norm dimension (L2-normalize) help / stabilise
vs the identical baseline on raw features?". Same params, same everything else.

PERMUTATION- & BAG-SIZE-INVARIANT: per-patch L2-norm is elementwise; softmax
attention + weighted mean are permutation-symmetric and size-agnostic.
DETERMINISTIC at inference (eval-mode dropout is identity; no sampling).

NUMERICAL SAFETY: ||f_i|| is divided with a .clamp(min=eps) floor so an
all-zero patch (or tiny norm) cannot produce NaN/inf — directly addressing the
NaN death of a prior candidate; real Virchow2 norms are ~6 so eps never bites.

PARAM COUNT (input_dim=1280, hidden=128, num_classes=1) — IDENTICAL to baseline
ABMIL because L2-normalization is parameter-free:
    bottleneck  Linear(1280,128)+b = 163,968
    attn_V      Linear(128,128)+b   =  16,512
    attn_U      Linear(128,128)+b   =  16,512
    attn_W      Linear(128,1)+b     =     129
    classifier  Linear(128,1)+b     =     129
    -------------------------------------------
    total                           = 197,250  (== baseline; capacity matched)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Baseline Ilse gated-attention MIL, optionally on L2-normalized patches.

    l2_normalize=True  -> a81 main (direction-only, magnitude removed).
    l2_normalize=False -> a82 ablation (== plain baseline on raw features).
    """

    def __init__(
        self,
        input_dim: int = 1280,
        num_classes: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        l2_normalize: bool = True,      # a81 main = ON; a82 ablation flips to False
        eps: float = 1e-6,              # norm floor -> no NaN on tiny/zero patches
    ) -> None:
        super().__init__()
        assert num_classes == 1, "scalar-regression head only (num_classes=1)."
        self.l2_normalize = bool(l2_normalize)
        self.eps = float(eps)

        # --- Baseline-identical gated-attention bottleneck (simple_mil idiom) ---
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
        metrics: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        # features: ONE bag [N, input_dim] (single-bag, batch_size=1 contract).
        # --- ACTIVE INGREDIENT: strip the grade-irrelevant magnitude -----------
        if self.l2_normalize:
            # Unit-direction per patch; eps floor keeps zero/tiny patches finite.
            features = features / features.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        # -----------------------------------------------------------------------

        # Baseline gated-attention weighted mean (operates on direction or raw).
        h = self.bottleneck(features)                       # [N, hidden]
        V = self.attention_V(h)                             # [N, hidden]
        U = self.attention_U(h)                             # [N, hidden]
        attn_scores = self.attention_W(V * U).squeeze(-1)   # [N]
        attention = F.softmax(attn_scores, dim=0)           # [N], perm-invariant
        z = torch.mm(attention.unsqueeze(0), h).squeeze(0)  # [hidden] weighted mean

        logits = self.classifier(z)                         # [1] RAW logit (no clamp)

        if return_attention:
            return logits, attention, None
        return logits, None, None


KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    hidden_dim=128,
    dropout=0.5,
    l2_normalize=True,      # a81 main = L2-normalize patches before aggregation
    eps=1e-6,
)
