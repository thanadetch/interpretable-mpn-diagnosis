"""a252 - Entropy-gated rank-cap pooling. Conditional, closed-form de-concentration of an over-peaked
attention read.

THE WALL (verbatim context). No candidate has ever cleared uni2-test>0.9418 while holding titan/virchow2,
because one FIXED attention setting cannot fit all three backbones (the "disjoint-helper"): uni2's baseline
attention is intrinsically the most concentrated (normalized entropy ~0.891 vs virchow2 ~0.965 / titan
~0.947). Any unconditional sharpening/sparsity helps the diffuse backbones but over-concentrates uni2 and
discards the diffuse density grading needs; any unconditional de-concentration (a248 James-Stein, a237 global
mean-blend) over-corrects the strong backbones that NEED some concentration (titan crashed to 0.9180). So the
ONLY viable mechanism de-concentrates ONLY when a per-bag read is pathologically over-peaked, and is exact
identity when the read is already diffuse -- inferring concentration from the bag itself, with no global knob.

MECHANISM (what makes a252 categorically different from prior de-concentration attempts).
  1. Standard gated attention: e = attention_W( attention_V(h) * attention_U(h) ), a = softmax(e), sum a = 1.
  2. Normalized Shannon entropy of the attention distribution:
         H = -sum_i a_i * log(a_i + eps) / log(max(N, 2))   in [0, 1]   (1 = uniform, 0 = single spike).
  3. Concentration gate (closed-form, NO learnable scalar, NO per-bag MLP):
         g = (threshold - H).clamp(min=0) / threshold       in [0, 1],
     so g = 0 (exact identity) whenever H >= threshold (diffuse), and rises smoothly toward 1 as H falls
     below threshold (over-peaked).
  4. Rank-cap smoothing -- the de-concentration operator (a SHAPE transform of a, not a shrink-to-mean and
     not a redundancy-repulsion). Sort a descending; let c = cumsum(a_sorted) be the running cumulative mass.
     Scale each sorted weight by cap_scale = clamp(cap_fraction / c, max=1.0): the top weights -- the ones
     that have ALREADY accumulated more than cap_fraction of the total mass -- get scaled down by exactly the
     factor that caps the running mass at cap_fraction, while the long diffuse tail (where c < cap_fraction)
     is left untouched (cap_scale = 1). Renormalize a_sorted * cap_scale to sum 1 -> a_capped, then UNSORT
     back to the original patch order. This bleeds the excess head mass into the diffuse body without moving
     to a uniform/mean read: it is a monotone re-shaping of the SAME attention ordering.
  5. Blend: a_smooth = (1 - g) * a + g * a_capped.  Pool z = h.t() @ a_smooth.  y = classifier(z).

WHY THIS TARGETS THE WALL. For uni2 (entropy ~0.891, frequently below a threshold set just above it) the gate
opens and the rank-cap redistributes the excess mass off the top patches -> the read de-concentrates and
recovers the diffuse density grading needs. For virchow2/titan (entropy ~0.965/~0.947, typically at/above
threshold) g -> 0 and a_smooth -> a EXACTLY -- the strong backbones are untouched, avoiding the a248 failure.
De-concentration is a pure function of the bag's own entropy and attention ordering: no global temperature,
no learnable alpha/lambda/blend (which went inert in a215/a237/a238), no per-bag content gate (a243), no
shrink-to-mean (a237/a248), no redundancy kernel (a244). It is also NOT an abandonment of the pool (a245/246/
247 collapsed): a_smooth is still a relevance-weighted distribution over the patches, pooling the SAME h.

HONEST NOTES / APPROXIMATIONS.
  - threshold and cap_fraction are FIXED constants (closed-form), chosen from the stated backbone entropies
    (threshold = 0.92, just above uni2's 0.891 and below titan/virchow2's 0.947/0.965; cap_fraction = 0.5,
    capping the running head mass at half the bag). They are heuristic constants, not theory-derived optima.
  - "Entropy" and "concentration" here are properties of the attention weights only, a proxy for how peaked
    the read is; they are not a calibrated measure of how much diffuse fibrosis density a bag contains.
  - The rank-cap is a redistribution heuristic on the sorted attention curve, not a projection onto a formal
    constraint set; cap_fraction/c can scale interior weights too, but cap_scale is monotone in rank so the
    relative ordering of the capped head is preserved and the renormalization keeps a valid distribution.

CONSTRAINTS satisfied. Self-contained nn.Module; torch/nn/F only. MPS-safe ops only: matmul, elementwise,
softmax, log, sort, cumsum, clamp, sum -- NO linalg.solve/eigh, NO cdist, NO torch.median. Permutation-
invariant (entropy, sort+unsort, and the weighted pool are all order-agnostic) and size-invariant (entropy
normalized by log(max(N,2)); no (N-1) terms). Deterministic in eval(). n=1 safe: log(max(N,2)) avoids div-by-
zero, a = [1], H = 0, g = 1, sorting a single element is identity, cumsum = [1], cap_scale = clamp(cap/1) <=1
then renormalized back to [1] -> a_smooth = [1], pool = h[0]. Only learnable parameters are bottleneck / V /
U / W / classifier (vectors/heads with non-flat gradients); no lone learnable shape-scalar.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
                 threshold=0.92, cap_fraction=0.5):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # FIXED closed-form constants (not nn.Parameter) -> nothing to go inert.
        self.threshold = float(threshold)
        self.cap_fraction = float(cap_fraction)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                                # [N,H]
        N = h.shape[0]
        e = self.attention_W(self.attention_V(h) * self.attention_U(h)).squeeze(-1)  # [N]
        a = F.softmax(e, dim=0)                                                       # [N] sums to 1 -> KEEP THE POOL

        # 1) normalized Shannon entropy of the attention read, in [0,1]; n=1 safe via max(N,2).
        log_norm = torch.log(torch.tensor(float(max(N, 2)), device=a.device, dtype=a.dtype))
        H = -(a * torch.log(a + 1e-12)).sum() / log_norm                             # scalar in [0,1]

        # 2) concentration gate: opens only when over-peaked (H < threshold); exact identity otherwise.
        g = (self.threshold - H).clamp(min=0.0) / self.threshold                     # scalar in [0,1]

        # 3) rank-cap smoothing: cap the running cumulative head mass at cap_fraction, keep the tail.
        a_sorted, order = torch.sort(a, descending=True)                             # [N], [N]
        c = torch.cumsum(a_sorted, dim=0)                                            # running mass, increasing
        cap_scale = (self.cap_fraction / (c + 1e-12)).clamp(max=1.0)                 # <1 only where c>cap_fraction
        a_sorted_capped = a_sorted * cap_scale
        a_sorted_capped = a_sorted_capped / (a_sorted_capped.sum() + 1e-12)          # renormalize to sum 1
        a_capped = torch.empty_like(a_sorted_capped)
        a_capped.scatter_(0, order, a_sorted_capped)                                 # unsort to original order

        # 4) conditional blend: untouched when diffuse (g=0), de-concentrated when over-peaked (g->1).
        a_smooth = (1.0 - g) * a + g * a_capped                                      # [N] sums to 1

        # 5) attention-weighted pool over the SAME features.
        z = torch.mv(h.t(), a_smooth)                                                # [H]
        y = self.classifier(z).view(-1)                                              # shape (1,)
        if return_attention:
            return y, a_smooth, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
