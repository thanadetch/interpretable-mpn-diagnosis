"""a259 - Hierarchical query refinement (Perceiver / Set-Transformer readout family).

THE WALL (verbatim context). ~276 prior aggregators have been tried; NONE pass the all-backbone gate.
The barrier is a val<->test QWK anti-correlation (~-0.95) on a tiny 214-ROI validation cohort that is
COHORT-driven and mechanism/capacity-independent. We are NOT promised any architecture breaks it. This
candidate contributes a GENUINELY-NEW mechanism FAMILY for completeness coverage of the search space: a
cross-attention readout with a SET of K learned queries that are ITERATIVELY REFINED by a shared MLP
conditioned on the current bag aggregate (a hierarchical/Perceiver-style read), then CONCATENATED into the
bag descriptor. This is categorically distinct from a single attention pool (the baseline) and from the
attention-shape/temperature/sparsity/de-concentration lane (entmax, size-temp, James-Stein shrink,
entropy-gated rank-cap/redundancy, variance-reg, robust-MAD, content-gate, mean-blend -- all done/failed).

MECHANISM.
  0. bottleneck: h = Dropout(ReLU(Linear(input_dim, hidden_dim)))(features) -> [N, 128]. ALL learnable
     content lives in vectors / heads / a shared MLP -- there are NO lone learnable shape-scalars (a215
     alpha, a238 temp, a237 lambda all went inert; avoided here).
  1. K learned query vectors  Q = nn.Parameter [K, 128]  (init randn * 128**-0.5). Each query is a distinct
     "read direction" over the bag, not a single attention head.
  2. Refinement loop, for t in range(n_iters):
       (a) scores = Q @ h.t() / sqrt(128)        -> [K, N]
           attn   = softmax(scores, dim=1)        -> [K, N]   (each query softmaxes over patches)
       (b) pool   = attn @ h                       -> [K, 128] (per-query relevance-weighted readout)
       (c) agg    = pool.mean(dim=0)               -> [128]    (size-invariant aggregate over the K query
                                                                 readouts -- a mean over QUERIES, never over a
                                                                 variable number of patches, so safe for any N)
       (d) refine each query (residual, shared MLP):
           Q = Q + query_refine( cat([Q, agg.expand(K, -1)], dim=1) )
           query_refine = Sequential(Linear(256,128), ReLU, Linear(128,128)) applied row-wise on [K, 256].
           The queries are nudged by the current global read so later iterations attend to complementary
           structure (hierarchical refinement); the shared MLP keeps capacity modest.
  3. Final cross-attention with the refined queries:
       scores = Q @ h.t() / sqrt(128); attn = softmax(scores, dim=1); pool = attn @ h -> [K, 128].
  4. bag = pool.reshape(-1) -> [K*128]   (CONCATENATE the K readouts -- a richer, still-pooled descriptor).
  5. y = classifier(bag).view(-1), classifier = Linear(K*128, num_classes) -> shape (num_classes,).

WHY THIS COUNTS AS A NEW FAMILY / WHY IT KEEPS A POOL. The bag descriptor is the CONCATENATION of K
relevance-weighted readouts (each pool = attn @ h is a valid weighted sum over the patches). It is NOT a
median/histogram/quantile-only read (a245/246/247 collapsed to ~random), NOT a single-query attention pool
(baseline), and NOT another attention-shape knob on one pool. The novel content is the iterative query
refinement conditioned on the bag aggregate -- a Perceiver/Set-Transformer-style read with a learned query
SET rather than one pooling vector.

HONEST NOTES / APPROXIMATIONS.
  - The "hierarchy" is shallow (n_iters=2): two refinement passes, not a deep stack. It is a light iterative
    readout, not a full Perceiver with cross+self-attention blocks (deliberately, to keep capacity near the
    197K baseline; historically capacity >> baseline overfits this 214-ROI cohort). K=4 queries, hidden 128.
  - agg = pool.mean(dim=0) is an UNWEIGHTED mean over the K query readouts. This is a design choice for
    size-invariance and determinism; it is not claimed to be an optimal fusion of the queries.
  - The refinement MLP is SHARED across queries and across iterations (no per-iteration parameters), again to
    cap capacity. There is no theoretical guarantee the refined queries become diverse / complementary; that
    is an empirical hope, not a proof. The model has no mechanism explicitly forcing query diversity.
  - There is no concept supervision: queries are learned purely from the regression signal (concept-free, no
    zero-shot text prompts, no bone/fibrosis labels). The descriptor is NOT norm-weighted (||h|| is
    grade-uninformative); attention uses scaled dot-product scores, not feature magnitude.

CONSTRAINTS satisfied. Self-contained nn.Module; torch / nn / F only; no external files, no new deps.
MPS-safe ops ONLY: matmul (@), elementwise, softmax, ReLU, Dropout, reshape/cat -- NO linalg.solve/eigh, NO
cdist, NO torch.median, no eigh, no solve. Permutation-invariant (softmax over patches + attn@h is an
order-agnostic weighted sum; agg/concat are over the K queries, which are fixed in order). Bag-size-invariant
(agg is a mean over K queries, never over N patches; no (N-1) terms, no std). Deterministic in eval() (only
Dropout is stochastic and is disabled in eval). n=1 safe: attn over 1 patch = softmax of a single score =
[1.0]; pool = h replicated per query [K,128]; agg = pool.mean over K queries (well-defined); the refinement
MLP is well-defined; final pool = h replicated; bag = h tiled K times; NO div-by-zero, NO std, NO (N-1).
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_dim=1280, num_classes=1, hidden_dim=128, dropout=0.5,
                 K=4, n_iters=2):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.K = int(K)
        self.n_iters = int(n_iters)
        self.scale = float(hidden_dim) ** -0.5

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        # K learned query VECTORS (non-flat gradients; not a lone learnable scalar).
        self.Q = nn.Parameter(torch.randn(self.K, hidden_dim) * (float(hidden_dim) ** -0.5))
        # Shared, row-wise refinement MLP: [K, 2*hidden] -> [K, hidden] (residual update for each query).
        self.query_refine = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim))
        self.classifier = nn.Linear(self.K * hidden_dim, num_classes)

    def forward(self, features, return_attention=False, metrics=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        h = self.bottleneck(features)                                    # [N, H]

        Q = self.Q                                                       # [K, H]
        # Iterative hierarchical refinement of the query set.
        for _ in range(self.n_iters):
            scores = (Q @ h.t()) * self.scale                            # [K, N]
            attn = F.softmax(scores, dim=1)                              # [K, N] (each query over patches)
            pool = attn @ h                                              # [K, H] per-query readout
            agg = pool.mean(dim=0)                                       # [H] size-invariant (mean over K)
            ctx = torch.cat([Q, agg.unsqueeze(0).expand(self.K, -1)], dim=1)  # [K, 2H]
            Q = Q + self.query_refine(ctx)                               # [K, H] residual update

        # Final cross-attention with the refined queries.
        scores = (Q @ h.t()) * self.scale                                # [K, N]
        attn = F.softmax(scores, dim=1)                                  # [K, N]
        pool = attn @ h                                                  # [K, H]

        bag = pool.reshape(-1)                                           # [K*H] concatenated readouts -> POOL
        y = self.classifier(bag).view(-1)                                # shape (num_classes,)
        if return_attention:
            return y, attn.mean(dim=0), None                             # [N] mean attention over queries
        return y, None, None


KWARGS = dict(input_dim=1280, num_classes=1)
