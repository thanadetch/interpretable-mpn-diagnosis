"""a315 — Power-mean (generalized-mean) attention pooling. A new aggregation FUNCTION.

DISCLOSURE: confirmatory fish under the user's explicit relaxed-criterion never-stop request. The
drop-in search is otherwise comprehensively closed (mechanism-class exhaustion + the one positive lead
a313 audited to seed-2 noise); GATE2 is not expected. This is a GENUINELY NEW aggregation (not a
normalization tweak, side-channel, ensemble, or capacity change), so it is a distinct test, not a
near-duplicate.

Mechanism: the baseline pools by a linear attention-weighted mean z = sum_i a_i h_i. a315 replaces it
with the element-wise WEIGHTED POWER-MEAN of the bottleneck features:
    z_j = ( sum_i a_i * h_ij^p )^(1/p) ,   p fixed (default 2).
This is well-defined because the bottleneck output is ReLU>=0, so h_ij >= 0. p=1 recovers the baseline
arithmetic mean; p>1 emphasises per-dimension "energy"/peaks (a density-of-activation read across the
bag); p->inf approaches the max. It aggregates the DISTRIBUTION of each feature over the bag rather
than just its mean — a different, density-aware summary aligned with the holistic-density grading
principle. p is FIXED (a learnable single scalar would be inert on this cohort — proven for alpha
a215 / lambda a312 / temperature a158), so a315 adds ZERO parameters over the baseline (head H->1) and
carries no extra capacity to overfit the 214-ROI val cohort. Attention scores are the unchanged Ilse
gated-attention softmax (uni2's optimum -> uni2-safe at the weighting level; only the combination rule
changes). No ||h|| weighting. Self-contained, concept-free, permutation/size-invariant, deterministic
eval, MPS-safe, no new deps.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, p=2.0):
        super().__init__()
        self.p = float(p)
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                                  # [N,H], >=0 (ReLU)
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)    # [N]
        a = F.softmax(e, dim=0)                                                         # unchanged softmax weights
        eps = 1e-6
        hp = (h + eps).pow(self.p)                                                      # [N,H], elementwise power
        z = torch.mv(hp.t(), a).clamp(min=eps).pow(1.0 / self.p)                        # weighted power-mean -> [H]
        y = self.classifier(z).view(-1)
        if return_attention:
            return y, a, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1, p=2.0)
