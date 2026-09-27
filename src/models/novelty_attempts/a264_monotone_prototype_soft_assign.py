"""a264 - Monotone-prototype soft-assignment pool. NEW family: ORDINAL MONOTONE-PROTOTYPE attention.
K learned grade prototypes are arranged on a STRICTLY MONOTONE ladder (cumulative-softplus offsets),
each patch is SOFTLY ASSIGNED across the ordered ladder (slot-style softmax OVER the K prototypes,
per patch), and the bag descriptor is the prototype-mixture vector plus an expected-ordinal-grade
read-out. This is NOT a single relevance-weighted mean of the patches, NOT an attention-shape/
temperature/sparsity tweak, NOT VLAD/Fisher, NOT capsule routing.

WHY THIS IS A CATEGORICALLY-DISTINCT FAMILY.
The exhausted lane shapes ONE relevance weight per patch and averages patch features (a single
first-moment pool, with all variation in HOW that weight is shaped). a264 does something structurally
different: it never produces a single per-patch relevance weight at all. Instead each patch is softly
distributed across K=4 prototypes via a softmax taken OVER THE PROTOTYPES (the slot/competition axis),
so for every patch the K assignment weights SUM TO ONE -- the patch "votes" for where it sits on an
ORDERED grade ladder. The prototypes themselves are constrained to be strictly increasing along the
ladder (P_0 < P_1 < P_2 < P_3 elementwise, by construction), which is what makes the K axis ordinal
rather than an unordered set of clusters. The bag is then summarised two ways that both depend on the
WHOLE distribution-over-the-ladder: (a) an expected-ordinal-grade scalar y_grade = sum_k k * w_k, and
(b) a pooled prototype-mixture vector z = sum_k w_k * P_k that the head reads.

DISTINCTNESS vs already-tried neighbours.
  - vs a259 / Perceiver multi-query cross-attn: those use generic UNORDERED learned queries with the
    softmax OVER PATCHES (queries gather a weighted mean of patches). a264 takes the softmax OVER
    PROTOTYPES per patch (competition across slots), and the prototypes carry an ENFORCED ordinal
    ordering. Opposite normalisation axis + monotone structure.
  - vs a251 consensus-CDF: a251 read a CDF over a single pooled vector. a264 keeps a per-patch
    soft-assignment to an ordered prototype ladder and reads the EXPECTED grade of that assignment.
  - vs a260 capsule routing: capsules do iterative agreement routing with squashing; a264 is a single
    non-iterative softmax-over-slots with NO routing iterations and NO squash nonlinearity, and the
    slots are an ORDERED ladder, not interchangeable capsules.
  - vs a93/a196 prototype/cluster pools: those soft-assign for clustering WITHOUT an enforced monotone
    ordinal ordering and WITHOUT the expected-ordinal-grade read-out.

MECHANISM (matmul / softmax / softplus / sum / elementwise only -- MPS-safe; no [N,K,D] tensor).
  1. h = bottleneck(features) = Dropout(ReLU(Linear(input_dim, hidden_dim)))                  -> [N,D]
  2. monotone prototype ladder. p_base in R^D (Parameter); deltas in R^{(K-1) x D} (Parameter).
     offsets_k = sum_{j<=k} softplus(deltas_j)  (cumulative, strictly nonneg); P_0 = p_base,
     P_k = p_base + offsets_{k-1}. Because softplus(.)>0, P is STRICTLY increasing elementwise along
     the ladder: P_0 < P_1 < P_2 < P_3. P is [K,D].
  3. patch-to-prototype scores e = (h @ P.t()) / sqrt(D)                                       -> [N,K]
  4. SLOT-style competition: alpha = softmax(e, dim=1) -- softmax OVER the K prototypes PER PATCH,
     so each row sums to 1 (the patch is softly assigned across the ordered ladder)              -> [N,K]
  5. size-invariant bag-level slot weights w_k = (1/N) sum_i alpha_i,k = mean over patches        -> [K]
     (mean, NOT sum -> bag-size-invariant; w sums to 1, a distribution over the grade ladder).
  6. expected ordinal grade y_grade = sum_k k * w_k  in [0, K-1] = [0,3]                          -> scalar
  7. pooled prototype-mixture descriptor z = sum_k w_k * P_k = w @ P  (the KEPT bag vector)      -> [D]
  8. readout: y_head = head(z).view(-1); final y = y_head + grade_gate * y_grade, where grade_gate
     is a learnable VECTOR-free... NO: grade_gate is implemented as a tiny learned linear blend so the
     model can use either signal. To avoid a lone inert shape-scalar, the blend weight on y_grade is the
     OUTPUT of a learned Linear(D->1) on z (content-dependent), NOT a bare scalar Parameter.            -> (1,)

HONEST NOTES / APPROXIMATIONS.
  - The "ordinal" structure is only an inductive bias: prototypes are forced monotone elementwise and
    the read-out is an expectation over integer ladder positions, but nothing forces the LEARNED bottleneck
    features to actually align with fibrosis severity. The ordering constrains the prototypes, not the data.
  - y_grade is a soft expectation in [0,3], not a calibrated grade; the SmoothL1 + round+clip lives in the
    trainer. The model just emits a continuous scalar.
  - The blend g(z) * y_grade lets the head route around y_grade if it is unhelpful; this is deliberate so
    the expected-grade branch cannot force a bad bias, but it does add a small Linear(D->1) head.
  - Concept-free: no zero-shot text prompts, no bone/fibrosis labels, no norm-weighting (||h|| is never a
    relevance signal; assignment comes only from h @ P.t()). The prototype ordering is data-agnostic.
  - The 1/sqrt(D) score scaling is a fixed temperature (standard attention scaling), not a learnable knob,
    so there is no lone learnable shape-scalar; all learnable content sits in p_base, deltas, bottleneck,
    the two heads -- vectors/matrices with non-flat gradients.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / torch.nn / torch.nn.functional only; no external
files, no new deps. MPS-safe ops ONLY: Linear, ReLU, Dropout, softplus, cumsum, softmax, matmul,
elementwise add/mul, sum -- NO linalg.solve/eigh/svd, NO cdist, NO torch.median. Permutation-invariant:
every per-patch contribution enters only through the mean over patches w = mean_i alpha_i (a sum/N), so
reordering patches leaves w, z, y_grade, y unchanged. Bag-size-invariant: w is a MEAN over patches (not a
sum); there is NO std, NO /(N-1); P, z, y_grade do not depend on N. Deterministic in eval() (Dropout off;
all other ops deterministic). n=1 SAFE: with one patch alpha is [1,K] summing to 1, w = alpha_1 (a valid
distribution), z = w @ P, y_grade in [0,3]; no division by (N-1), no eps-fragile op. Capacity is modest:
hidden_dim=128 bottleneck, K=4 prototypes (p_base [D], deltas [(K-1),D]), two small heads -- well under
197K params; no per-cluster D*D matrix.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5, K=4):
        super().__init__()
        assert K >= 2, "need at least 2 prototypes to form an ordinal ladder"
        self.K = K
        self.hidden_dim = hidden_dim
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # monotone prototype ladder: base + cumulative-softplus deltas (built in forward)
        self.p_base = nn.Parameter(torch.randn(hidden_dim) * (hidden_dim ** -0.5))   # P_0 anchor [D]
        # init deltas small (softplus(small) -> small positive) so the ladder starts gently separated.
        self.deltas = nn.Parameter(torch.randn(K - 1, hidden_dim) * 0.01)            # [(K-1),D]
        # integer ladder positions 0..K-1 for the expected-ordinal-grade read-out
        self.register_buffer("ladder", torch.arange(K, dtype=torch.float32))          # [K]
        self.head = nn.Linear(hidden_dim, num_classes)        # reads the prototype-mixture vector z
        self.grade_blend = nn.Linear(hidden_dim, 1)           # content-dependent blend weight on y_grade

    def _prototypes(self) -> torch.Tensor:
        # strictly monotone ladder: P_0 = p_base; P_k = p_base + cumsum_{j<=k-1} softplus(delta_j)
        offsets = torch.cumsum(F.softplus(self.deltas), dim=0)          # [(K-1),D], strictly increasing
        P = torch.cat([self.p_base.unsqueeze(0), self.p_base.unsqueeze(0) + offsets], dim=0)  # [K,D]
        return P

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                   # [N,D]
        P = self._prototypes()                                          # [K,D] strictly monotone ladder
        e = (h @ P.t()) / (self.hidden_dim ** 0.5)                      # [N,K] patch-to-prototype scores
        alpha = F.softmax(e, dim=1)                                     # [N,K] softmax OVER prototypes per patch
        w = alpha.mean(dim=0)                                           # [K] size-invariant slot distribution
        y_grade = (w * self.ladder).sum()                              # scalar expected ordinal grade in [0,K-1]
        z = w @ P                                                       # [D] prototype-mixture bag descriptor
        y_head = self.head(z).view(-1)                                  # (num_classes,)
        g = self.grade_blend(z).view(-1)                               # (1,) content-dependent blend weight
        y = (y_head + g * y_grade).view(-1)                            # (1,) final scalar regression output
        if return_attention:
            return y, alpha, None                                       # [N,K] per-patch ladder assignment
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
