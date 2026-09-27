"""a267 - Slot-Attention pooling (Locatello et al. 2020). NEW family: competitive slots.

THE WALL (verbatim context). ~287 prior aggregators tried; NONE pass the all-backbone gate. The barrier is
a val<->test QWK anti-correlation (~-0.95) on a tiny 214-ROI validation cohort, COHORT-driven and
mechanism/capacity-independent. a267 is a COMPLETENESS-COVERAGE contribution: a categorically-distinct
pooling family (Slot Attention) not yet implemented (the ideation agent for it failed to return structured
output; the idea was never rejected).

WHY THIS IS A DISTINCT FAMILY. Standard gated attention (the baseline) and almost every prior candidate
normalize a relevance distribution OVER PATCHES (softmax over the N patch dimension) and pool one weighted
mean. Slot Attention instead introduces K learnable SLOTS that COMPETE for patches: the attention is a
softmax NORMALIZED OVER THE SLOTS (each patch distributes its mass across the K slots, the slots compete),
then each slot is updated as a patch-normalized weighted mean of the values it won. Iterating this a few
times yields K slot vectors that partition/explain the bag; the bag descriptor is read from the slots. The
"softmax over slots" competition is the categorically-new mechanism -- distinct from capsule routing (a260,
agreement-iterated votes + squash), from Set-Transformer ISAB (a261, inducing-point patch<->patch
self-attention), from Perceiver multi-query (a259, independent query readouts, NO inter-slot competition),
and from plain over-patch attention.

MECHANISM (per bag).
  1. h   = Dropout(ReLU(Linear(input_dim, D)))(features)            -> [N, D]
  2. h_n = LayerNorm(h)                                             -> [N, D]
     k = to_k(h_n), v = to_v(h_n)                                   -> [N, D], [N, D]
  3. slots = learnable init S0 (K vectors, FIXED learnable -> deterministic; no Gaussian sampling).
  4. For t in 1..T (T=3):
       q          = to_q(LayerNorm(slots))                          -> [K, D]
       logits     = (k @ q.t()) / sqrt(D)                           -> [N, K]
       attn       = softmax(logits, dim=1)   # OVER SLOTS (competition; each patch's row sums to 1)  [N, K]
       w          = attn / (attn.sum(dim=0, keepdim=True) + eps)    # normalize over patches -> weighted mean
       updates    = w.t() @ v                                       -> [K, D]
       slots      = LayerNorm(slots + updates)   # residual update (GRU replaced by residual+LN; MPS-safe,
                                                 #   deterministic; a standard slot-attention simplification)
  5. z = slots.mean(dim=0)                                          -> [D]  (permutation-invariant over slots)
  6. y = classifier(z).view(-1)                                     -> (1,)

WHY IT KEEPS A POOLED VECTOR. Each slot is itself a patch-weighted mean of values (a valid pooled vector);
z is the mean of the K slots. This is NOT a median/histogram/quantile-only read (which collapsed in
a245/246/247) -- it is a structured set of weighted means.

HARD-CONSTRAINT compliance. Self-contained nn.Module; torch/nn/F only; no external files/deps. MPS-safe ops
ONLY: Linear, ReLU, Dropout, LayerNorm, matmul, softmax, sum, mean, sqrt -- NO linalg.solve/eigh/svd, NO
cdist, NO torch.median, NO GRU (the recurrent update is a residual+LayerNorm, avoiding MPS RNN issues).
Permutation-invariant over patches: k/v are per-patch maps, the over-slot softmax + over-patch normalized
weighted mean + slot mean are all symmetric in the patch index. Bag-size-invariant: w is normalized over
patches (a weighted mean, no /N, no /(N-1), no std). Deterministic in eval(): slot init is a fixed learnable
Parameter (NOT randn-sampled per forward), dropout is disabled in eval(), T is a fixed integer. n=1 safe:
N=1 -> attn is [1,K] softmax over slots, w = attn (sum over the single patch), updates = w.t() @ v works;
no division by zero/std/(N-1). Capacity modest (~214K, comparable to the 197K baseline). Learnable content
lives in matrices/heads (bottleneck, to_q/k/v, slot init [K,D], classifier) -- NO lone learnable shape-scalar
(T and K are fixed hyperparameters), so nothing can go inert. Concept-free; NOT norm-weighted.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, n_slots=4, n_iters=3, dropout=0.5):
        super().__init__()
        self.n_slots = int(n_slots)
        self.n_iters = int(n_iters)
        self.scale = hidden_dim ** -0.5
        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.norm_in = nn.LayerNorm(hidden_dim)
        self.norm_slots = nn.LayerNorm(hidden_dim)
        self.norm_upd = nn.LayerNorm(hidden_dim)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Fixed-learnable slot initialization (deterministic; no per-forward sampling).
        self.slots_init = nn.Parameter(torch.randn(n_slots, hidden_dim) * (hidden_dim ** -0.5))
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        eps = 1e-8
        h = self.bottleneck(features)                                  # [N, D]
        h_n = self.norm_in(h)
        k = self.to_k(h_n)                                             # [N, D]
        v = self.to_v(h_n)                                             # [N, D]

        slots = self.slots_init                                       # [K, D]
        attn = None
        for _ in range(self.n_iters):
            q = self.to_q(self.norm_slots(slots))                     # [K, D]
            logits = (k @ q.t()) * self.scale                         # [N, K]
            attn = F.softmax(logits, dim=1)                           # over SLOTS (competition) [N, K]
            w = attn / (attn.sum(dim=0, keepdim=True) + eps)          # normalize over patches -> weighted mean
            updates = w.t() @ v                                       # [K, D]
            slots = self.norm_upd(slots + updates)                    # residual update (MPS-safe, deterministic)

        z = slots.mean(dim=0)                                         # [D] permutation-invariant over slots
        y = self.classifier(z).view(-1)                              # (1,)
        if return_attention:
            # per-patch saliency = total mass each patch sent to any slot, normalized to a distribution
            sal = attn.sum(dim=1)
            sal = sal / (sal.sum() + eps)
            return y, sal, None
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
