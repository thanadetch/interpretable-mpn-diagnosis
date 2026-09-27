"""a43 — rank-norm soft-mean port to the Virchow2 + ABMIL baseline.

Philosophy bucket: attention_replace_parameter_free

Background / why this exists
----------------------------
On the legacy UNI2 + mean_pool baseline (val_qwk 0.8090 / test_qwk 0.9129),
the only novelty that ever beat both gates was `a25_rank_norm_softmean`
(val_qwk 0.8185 / test_qwk 0.9235, +0.0095 / +0.0106 over baseline).
The new locked baseline is Virchow2 + ABMIL (val_qwk 0.8182 /
test_qwk 0.9476), and a25 was never re-evaluated against it. After 40
novelty runs (a01–a40, see leaderboard_v2.csv) failed to beat the new
gates, the rank-norm idea is the best remaining candidate that has NOT
been tested on the new setup.

Hypothesis (H21_rank_norm_softmean_virchow2)
--------------------------------------------
Patch L2 norm in pathology foundation embeddings tracks "tissue
salience" (low-norm patches tend to be near-empty / low-content). A
parameter-free `softmax(-rank / τ)` weighting therefore down-weights
likely-empty patches without needing a learnable attention head, which
should reduce overfitting on the small 30-patient train set. The legacy
a25 confirmed this on UNI2 features; Virchow2 patch norms have wider
dynamic range, so the rank-based weighting may translate (or may not).

Mechanism (identical to the attached legacy a25)
------------------------------------------------
1. Input features [N, D=1280] (Virchow2). No bottleneck.
2. Per-patch norm  n_i = ||h_i||_2.
3. Rank patches by norm descending → ranks ∈ {0, …, N-1}.
4. Weights      w_i = softmax(-rank_i / τ) over patches, τ = 8.
5. Bag rep      b   = Σ_i w_i · h_i      ∈ R^D
6. Head         ŷ   = Linear(D, 1) ( + Dropout(0.5) on b ), clamp to [0, 3].

Role: MAIN of the H21 family. Bit-for-bit port of legacy a25 to
Virchow2 (only `input_dim` differs: 1536 → 1280). Returns the standard
3-tuple `(logits, attention, None)` so the existing trainer code path
(`logits, _, _ = model(features)`) works unchanged.

Pair with
---------
a44_rank_norm_softmean_bottleneck (ablation companion). a44 adds the
ABMIL bottleneck `Linear(1280, 128) + ReLU + Dropout(0.5)` in
front, then applies the same rank-norm softmax-mean over the
bottlenecked features. The a43 ↔ a44 contrast isolates whether
"no bottleneck" is part of the active ingredient of legacy a25, or
whether the rank-norm weighting alone carries the win.

Kill criterion
--------------
Family-level: if BOTH a43 and a44 have val_qwk < 0.80 at seed=2,
declare H21 dead under the Virchow2 baseline and stop the search line.

Param count (input_dim=1280)
----------------------------
    classifier  Linear(1280, 1) + bias  = 1,281
    --------------------------------------------
    total trainable                     = 1,281
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 1,
        dropout: float = 0.5,
        tau: float = 8.0,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        assert num_classes == 1, "a43 is scalar-regression only (num_classes=1)."
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )
        self.tau = tau
        self.clamp_output = clamp_output

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,  # accepted for trainer signature parity
        metrics: Optional[dict] = None,  # ignored
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)  # [1, N, D]
        B, N, D = features.shape

        # 1) per-patch L2 norm as a salience proxy
        norms = features.norm(dim=-1)  # [B, N]

        # 2) rank patches by norm descending (0 = highest norm)
        order = norms.argsort(dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(
            1, order, torch.arange(N, device=features.device).expand(B, N)
        )

        # 3) parameter-free softmax over negative ranks
        w = torch.softmax(-ranks.float() / self.tau, dim=1)  # [B, N]

        # 4) weighted mean of raw features
        bag = (w.unsqueeze(-1) * features).sum(dim=1)  # [B, D]

        # 5) head
        y = self.classifier(bag)  # [B, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if squeeze:
            y = y.squeeze(0)  # [1]
            w = w.squeeze(0)  # [N]
        return y, w, None


# input_dim is overridden by the trainer to match the selected backbone
# (virchow2 → 1280). Leaving 1280 here makes the default explicit.
KWARGS = dict(
    input_dim=1280,
    num_classes=1,
    dropout=0.5,
    tau=8.0,
    clamp_output=True,
)

